# Importing the Keras libraries and packages
import time
from keras.models import load_model
from keras.preprocessing import image
import glob
import numpy as np
import json
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import sys

# GET DATA

# get data from motionflow dataset

motionflow_files = [
    f for f in glob.glob("data/motionflow/**/*.jpg", recursive=True)
]

imdb_crop_part1_files = [
    f for f in glob.glob("data/crop_part1/**/*.jpg", recursive=True)
]

races = ["indian", "hispanic", "caucasian", "asian"]


def age_to_age_category(age):
    if age < 18:
        return "18minus"
    elif age <= 24:
        return "18to24"
    elif age <= 34:
        return "25to34"
    elif age <= 44:
        return "35to44"
    elif age <= 54:
        return "45to54"
    elif age <= 64:
        return "55to64"
    elif age <= 75:
        return "65to75"
    else:
        return "75plus"


def get_motionflow_row(f):
    path = f.split("/")
    race = path[-2]

    if race not in races:
        raise Exception(f)

    filename = ".".join(path[-1].split(".")[:-1])

    parts = filename.split("_")

    gender = parts[-1].strip()

    if gender != "M" and gender != "F":
        print("parts")
        print(parts)
        raise Exception(f)

    age = int(parts[0].split(".")[0])

    age_category = age_to_age_category(age)

    return {
        "age": age,
        "age_category": age_category,
        "gender": gender,
        "race": race,
        "filename": f,
    }


failed_to_parse_imdb_rows = []


def get_imdb_row(f):
    try:
        path = f.split("/")

        filename = path[-1].split(".")[0]

        parts = filename.split("_")

        gender = parts[-1].strip()

        gender_indices = {"1": "F", "0": "M"}

        race_indices = {
            "0": "caucasian",
            "1": "african-american",
            "2": "asian",
            "3": "indian",
            "4": "hispanic",
        }

        gender = gender_indices[parts[1]]
        race = race_indices[parts[2]]
        age = int(parts[0])

        age_category = age_to_age_category(age)

        return {
            "age": age,
            "age_category": age_category,
            "gender": gender,
            "race": race,
            "filename": f,
        }
    except Exception as e:
        print("error at filepath: " + f)
        failed_to_parse_imdb_rows.append(f)
        return None


def mapMaybe(f, xs):
    return list(filter(lambda x: x is not None, map(f, xs)))


rows = mapMaybe(get_motionflow_row, motionflow_files) + mapMaybe(
    get_imdb_row, imdb_crop_part1_files
)

print("failed to parse imdb: " + str(len(failed_to_parse_imdb_rows)))

dataframe = pd.DataFrame(rows)

# get data from crop_part1

imdb_files = [f for f in glob.glob("data/crop_part1/**/*.jpg", recursive=True)]

# BUILD CNN MODEL

if "--retrain" in sys.argv:
    print("retraining model")

    def make_cnn(class_mode, output_dim):
        classifier = Sequential()

        # Step 1 - Convolution
        classifier.add(
            Conv2D(32, 3, 3, input_shape=(64, 64, 3), activation="relu")
        )

        # Step 2 - Pooling
        classifier.add(MaxPooling2D(pool_size=(2, 2)))

        # Adding a second convolutional layer
        classifier.add(Conv2D(32, 3, 3, activation="relu"))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))

        # Step 3 - Flattening
        classifier.add(Flatten())

        # Step 4 - Full connection
        classifier.add(Dense(output_dim=128, activation="relu"))

        if output_dim == 1:
            last_activation = "sigmoid"
        else:
            last_activation = "softmax"

        classifier.add(
            Dense(output_dim=output_dim, activation=last_activation)
        )

        if class_mode == "binary":
            loss = "binary_crossentropy"
        elif class_mode == "categorical":
            loss = "categorical_crossentropy"
        elif class_mode == "continuous":
            loss = "mse"
        else:
            print(class_mode)
            raise ValueError("unrecognised class_mode")

        # Compiling the CNN
        classifier.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

        return classifier

    # FIT MODEL AND SAVE MODEL AND INDICES
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )

    def fit_and_save_model(target, class_mode):
        df = dataframe

        if class_mode == "categorical":
            output_dim = len(np.unique(df[target]))
        else:
            output_dim = 1

        classifier = make_cnn(class_mode, output_dim)

        if class_mode == "binary":
            class_mode = "binary"
        elif class_mode == "categorical":
            class_mode = "categorical"
        elif class_mode == "continuous":
            class_mode = "mse"  # TODO
        else:
            print(class_mode)
            raise ValueError("unrecognised class_mode")

        training_set = train_datagen.flow_from_dataframe(
            df,
            directory=None,
            x_col="filename",
            y_col=target,
            target_size=(64, 64),
            class_mode=class_mode,
            batch_size=32,
            subset="training",
        )

        test_set = train_datagen.flow_from_dataframe(
            df,
            directory=None,
            x_col="filename",
            y_col=target,
            target_size=(64, 64),
            class_mode=class_mode,
            batch_size=32,
            subset="validation",
        )

        print(type(test_set))

        classifier.fit_generator(
            training_set,
            samples_per_epoch=10000,
            epochs=10,
            validation_data=test_set,
            validation_steps=1000,
        )

        classifier.save(target + "_model.h5")
        with open(target + "_class_indices.json", "w") as json_file:
            json.dump(training_set.class_indices, json_file)

        return classifier, training_set, test_set

    fit_and_save_model("race", "categorical")

    fit_and_save_model("gender", "binary")

    fit_and_save_model("age_category", "categorical")

# PREDICT VALUES DATA TYPES
print("predicting")
age_category_model = load_model("./age_category_model.h5")
gender_model = load_model("./gender_model.h5")
race_model = load_model("./race_model.h5")

with open("age_category_class_indices.json", "r") as json_file:
    age_category_indices = {v: k for k, v in json.load(json_file).items()}

with open("gender_class_indices.json", "r") as json_file:
    gender_indices = {v: k for k, v in json.load(json_file).items()}

with open("race_class_indices.json", "r") as json_file:
    race_indices = {v: k for k, v in json.load(json_file).items()}


def get_image(filepath):
    img = image.load_img(filepath, target_size=(64, 64))
    arr = image.img_to_array(img)
    return np.expand_dims(arr, axis=0)


def predict_age_category(filepath):
    raw_prediction = age_category_model.predict(get_image(filepath))
    values = list(raw_prediction[0])
    max_i = values.index(max(values))
    return age_category_indices[max_i]


def predict_race(filepath):
    raw_prediction = race_model.predict(get_image(filepath))
    values = list(raw_prediction[0])
    max_i = values.index(max(values))
    return race_indices[max_i]


def predict_gender(filepath):
    raw_prediction = gender_model.predict(get_image(filepath))
    return gender_indices[round(raw_prediction[0, 0])]


# TEST SINGLE
test_file = "data/motionflow/caucasian/49.1_M.jpg"

start = time.time()

print("single results")
print(predict_gender(test_file))
print(predict_age_category(test_file))
print(predict_race(test_file))

end = time.time()
print("timing")
print((end - start) / 3)

# OVERALL RESULTS

dataframe["predicted_age_category"] = dataframe.apply(
    lambda r: predict_age_category(r.filename), axis="columns"
)
dataframe["predicted_gender"] = dataframe.apply(
    lambda r: predict_gender(r.filename), axis="columns"
)
dataframe["predicted_race"] = dataframe.apply(
    lambda r: predict_race(r.filename), axis="columns"
)

results = {}
for col in ["age_category", "gender", "race"]:
    correct = (dataframe[col] == dataframe["predicted_" + col]).sum()
    incorrect = (dataframe[col] != dataframe["predicted_" + col]).sum()
    results[col] = {
        "correct": int(correct),
        "incorrect": int(incorrect),
        "accuracy": float(correct / (correct + incorrect)),
    }

print("all results")
print(json.dumps(results, indent=4))
