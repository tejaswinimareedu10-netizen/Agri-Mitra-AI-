# ============================================================
# AGRIMITRA AI
# Flask Backend
#
# Features:
# 1. Crop Recommendation
# 2. Crop Yield Prediction
# 3. Real-Time Weather
# 4. Indian Location Search
# 5. Agricultural Weather Alerts
# 6. Chatbot API
#
# No IoT required
# ============================================================

import os
import pickle
import traceback
import requests

import numpy as np
import pandas as pd
import joblib

from flask import (
    Flask,
    request,
    jsonify,
    render_template
)

from flask_cors import CORS

from tensorflow import keras
from xgboost import XGBRegressor


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ============================================================
# MODEL FILES
# ============================================================

CROP_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "AgriMitra_Crop_Recommendation_Model.keras"
)

CROP_SCALER_PATH = os.path.join(
    MODEL_DIR,
    "AgriMitra_Crop_Scaler.pkl"
)

CROP_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "AgriMitra_Crop_LabelEncoder.pkl"
)


YIELD_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "AgriMitra_XGBoost_Yield_Model (1).json"
)

YIELD_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "AgriMitra_TargetEncoder (1).pkl"
)


# ============================================================
# WEATHER API
# ============================================================

GEOCODING_URL = (
    "https://photon.komoot.io/api/"
)

WEATHER_URL = (
    "https://api.open-meteo.com/v1/forecast"
)


# ============================================================
# GLOBAL MODELS
# ============================================================

crop_model = None
crop_scaler = None
crop_label_encoder = None

yield_model = None
yield_encoder = None


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    global crop_model
    global crop_scaler
    global crop_label_encoder

    global yield_model
    global yield_encoder

    print("=" * 70)
    print("AGRIMITRA AI - LOADING MODELS")
    print("=" * 70)

    # --------------------------------------------------------
    # CROP MODEL
    # --------------------------------------------------------

    if not os.path.exists(
        CROP_MODEL_PATH
    ):
        raise FileNotFoundError(
            f"Crop model not found:\n{CROP_MODEL_PATH}"
        )

    if not os.path.exists(
        CROP_SCALER_PATH
    ):
        raise FileNotFoundError(
            f"Crop scaler not found:\n{CROP_SCALER_PATH}"
        )

    if not os.path.exists(
        CROP_ENCODER_PATH
    ):
        raise FileNotFoundError(
            f"Crop label encoder not found:\n{CROP_ENCODER_PATH}"
        )


    crop_model = keras.models.load_model(
        CROP_MODEL_PATH
    )

    crop_scaler = joblib.load(
        CROP_SCALER_PATH
    )

    crop_label_encoder = joblib.load(
        CROP_ENCODER_PATH
    )


    print(
        "Crop recommendation model loaded."
    )

    print(
        "Crop classes:",
        len(
            crop_label_encoder.classes_
        )
    )


    # --------------------------------------------------------
    # YIELD MODEL
    # --------------------------------------------------------

    if not os.path.exists(
        YIELD_MODEL_PATH
    ):
        raise FileNotFoundError(
            f"Yield model not found:\n{YIELD_MODEL_PATH}"
        )

    if not os.path.exists(
        YIELD_ENCODER_PATH
    ):
        raise FileNotFoundError(
            f"Yield encoder not found:\n{YIELD_ENCODER_PATH}"
        )


    yield_model = XGBRegressor()

    yield_model.load_model(
        YIELD_MODEL_PATH
    )


    with open(
        YIELD_ENCODER_PATH,
        "rb"
    ) as file:

        yield_encoder = pickle.load(
            file
        )


    print(
        "Yield XGBoost model loaded."
    )

    print(
        "Yield target encoder loaded."
    )

    print("=" * 70)
    print("ALL MODELS LOADED SUCCESSFULLY")
    print("=" * 70)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status": "ok",

        "crop_model_loaded":
            crop_model is not None,

        "yield_model_loaded":
            yield_model is not None,

        "weather":
            "API available",

        "location_search":
            "API available",

        "chatbot":
            "API available"

    })


# ============================================================
# INDIA LOCATION SEARCH
# ============================================================

@app.route(
    "/api/weather/locations",
    methods=["GET"]
)
def weather_locations():

    try:

        location = request.args.get(
            "location",
            ""
        ).strip()


        if not location:

            return jsonify({

                "success": False,

                "message":
                    "Please enter a location."

            }), 400


        params = {

            "q":
                location,

            "limit":
                10,

            "lang":
                "en",

            "lat":
                20.5937,

            "lon":
                78.9629

        }


        response = requests.get(

            GEOCODING_URL,

            params=params,

            headers={
                "User-Agent":
                    "AgriMitraAI/1.0"
            },

            timeout=15

        )


        response.raise_for_status()


        data = response.json()


        results = data.get(
            "features",
            []
        )


        locations = []


        for result in results:

            props = result.get(
                "properties",
                {}
            )


            country_code = str(
                props.get(
                    "countrycode",
                    ""
                )
            ).upper()


            if (
                country_code
                and country_code != "IN"
            ):
                continue


            coords = result.get(
                "geometry",
                {}
            ).get(
                "coordinates",
                [None, None]
            )


            locations.append({

                "name":
                    props.get(
                        "name",
                        "Unknown"
                    ),

                "latitude":
                    coords[1],

                "longitude":
                    coords[0],

                "state":
                    props.get(
                        "state",
                        ""
                    ),

                "district":
                    props.get(
                        "county",
                        props.get(
                            "district",
                            ""
                        )
                    ),

                "subdistrict":
                    props.get(
                        "district",
                        ""
                    ),

                "country":
                    props.get(
                        "country",
                        "India"
                    ),

                "country_code":
                    "IN",

                "timezone":
                    "Asia/Kolkata",

                "population":
                    props.get(
                        "population"
                    ),

                "feature_code":
                    props.get(
                        "osm_value",
                        ""
                    )

            })


        if not locations:

            return jsonify({

                "success": False,

                "message":
                    (
                        f"No locations found "
                        f"for '{location}'. "
                        f"Try adding the district "
                        f"or state."
                    )

            }), 404


        return jsonify({

            "success": True,

            "locations":
                locations

        })


    except requests.RequestException as e:

        print(
            "Location API error:"
        )

        print(
            traceback.format_exc()
        )

        return jsonify({

            "success": False,

            "message":
                "Location service is unavailable.",

            "details":
                str(e)

        }), 503


    except Exception as e:

        print(
            traceback.format_exc()
        )

        return jsonify({

            "success": False,

            "message":
                "Location search failed.",

            "details":
                str(e)

        }), 500

# ============================================================
# WEATHER DESCRIPTION
# ============================================================

def weather_description(
    code
):

    descriptions = {

        0: "Clear sky",

        1: "Mainly clear",

        2: "Partly cloudy",

        3: "Overcast",

        45: "Fog",

        48: "Rime fog",

        51: "Light drizzle",

        53: "Moderate drizzle",

        55: "Dense drizzle",

        56: "Light freezing drizzle",

        57: "Dense freezing drizzle",

        61: "Slight rain",

        63: "Moderate rain",

        65: "Heavy rain",

        66: "Light freezing rain",

        67: "Heavy freezing rain",

        71: "Slight snowfall",

        73: "Moderate snowfall",

        75: "Heavy snowfall",

        77: "Snow grains",

        80: "Slight rain showers",

        81: "Moderate rain showers",

        82: "Violent rain showers",

        85: "Slight snow showers",

        86: "Heavy snow showers",

        95: "Thunderstorm",

        96: "Thunderstorm with hail",

        99: "Heavy thunderstorm with hail"

    }

    return descriptions.get(
        code,
        "Unknown"
    )


# ============================================================
# GENERATE AGRICULTURAL ALERTS
# ============================================================

def generate_agriculture_alerts(
    current,
    hourly
):

    alerts = []


    temperature = current.get(
        "temperature_2m",
        0
    ) or 0


    humidity = current.get(
        "relative_humidity_2m",
        0
    ) or 0


    wind_speed = current.get(
        "wind_speed_10m",
        0
    ) or 0


    wind_gust = current.get(
        "wind_gusts_10m",
        0
    ) or 0


    precipitation_probability = (
        hourly.get(
            "precipitation_probability",
            []
        )
    )


    precipitation = (
        hourly.get(
            "precipitation",
            []
        )
    )


    next_24_probability = max(
        precipitation_probability[:24],
        default=0
    )


    next_24_rain = sum(
        precipitation[:24]
    )


    # --------------------------------------------------------
    # RAIN
    # --------------------------------------------------------

    if (
        next_24_probability >= 70
        or next_24_rain >= 5
    ):

        alerts.append({

            "type":
                "rain",

            "severity":
                "high",

            "title":
                "Rain Expected",

            "message":
                (
                    f"Rain is likely within "
                    f"the next 24 hours. "
                    f"Expected precipitation "
                    f"is approximately "
                    f"{next_24_rain:.1f} mm. "
                    f"Consider postponing "
                    f"irrigation if soil "
                    f"moisture is adequate."
                )

        })


    elif next_24_probability >= 40:

        alerts.append({

            "type":
                "rain",

            "severity":
                "medium",

            "title":
                "Possible Rain",

            "message":
                (
                    "Moderate rainfall "
                    "probability detected. "
                    "Check the forecast "
                    "before irrigation."
                )

        })


    # --------------------------------------------------------
    # EXTREME HEAT
    # --------------------------------------------------------

    if temperature >= 40:

        alerts.append({

            "type":
                "heat",

            "severity":
                "high",

            "title":
                "Extreme Heat Alert",

            "message":
                (
                    "Very high temperature "
                    "may cause crop heat "
                    "stress. Monitor soil "
                    "moisture carefully."
                )

        })


    elif temperature >= 35:

        alerts.append({

            "type":
                "heat",

            "severity":
                "medium",

            "title":
                "High Temperature",

            "message":
                (
                    "High temperature "
                    "detected. Monitor "
                    "soil moisture "
                    "carefully."
                )

        })


    # --------------------------------------------------------
    # LOW HUMIDITY
    # --------------------------------------------------------

    if humidity <= 30:

        alerts.append({

            "type":
                "dryness",

            "severity":
                "medium",

            "title":
                "Dry Atmospheric Conditions",

            "message":
                (
                    "Low humidity may "
                    "increase evaporation "
                    "and crop water demand."
                )

        })


    # --------------------------------------------------------
    # STRONG WIND
    # --------------------------------------------------------

    if (
        wind_speed >= 40
        or wind_gust >= 50
    ):

        alerts.append({

            "type":
                "wind",

            "severity":
                "high",

            "title":
                "Strong Wind Alert",

            "message":
                (
                    "Strong winds may "
                    "increase water loss. "
                    "Avoid unnecessary "
                    "irrigation during "
                    "strong winds."
                )

        })


    # --------------------------------------------------------
    # DRY WEATHER
    # --------------------------------------------------------

    if (
        next_24_probability < 20
        and next_24_rain < 1
        and temperature >= 30
    ):

        alerts.append({

            "type":
                "dry",

            "severity":
                "medium",

            "title":
                "Dry Weather",

            "message":
                (
                    "Little rainfall is "
                    "expected while "
                    "temperature is "
                    "relatively high. "
                    "Irrigation demand "
                    "may increase."
                )

        })


    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    if not alerts:

        alerts.append({

            "type":
                "normal",

            "severity":
                "low",

            "title":
                "Normal Weather Conditions",

            "message":
                (
                    "No major agricultural "
                    "weather risk was "
                    "detected."
                )

        })


    return alerts


# ============================================================
# REAL-TIME WEATHER
# ============================================================

@app.route(
    "/api/weather",
    methods=["POST"]
)
def weather():

    try:

        data = (
            request.get_json(
                silent=True
            )
            or {}
        )


        latitude = data.get(
            "latitude"
        )

        longitude = data.get(
            "longitude"
        )


        location_data = data.get(
            "location"
        )


        # ----------------------------------------------------
        # VALIDATE COORDINATES
        # ----------------------------------------------------

        if (
            latitude is None
            or longitude is None
        ):

            return jsonify({

                "success": False,

                "error":
                    (
                        "Latitude and "
                        "longitude are required."
                    )

            }), 400


        latitude = float(
            latitude
        )

        longitude = float(
            longitude
        )


        # ----------------------------------------------------
        # LOCATION INFORMATION
        # ----------------------------------------------------

        if isinstance(
            location_data,
            dict
        ):

            location_name = (
                location_data.get(
                    "name",
                    "Selected location"
                )
            )

            state = (
                location_data.get(
                    "state",
                    ""
                )
            )

            district = (
                location_data.get(
                    "district",
                    ""
                )
            )

            country = (
                location_data.get(
                    "country",
                    "India"
                )
            )

        else:

            location_name = str(
                location_data
                or "Selected location"
            )

            state = ""
            district = ""
            country = "India"


        # ----------------------------------------------------
        # WEATHER PARAMETERS
        # ----------------------------------------------------

        params = {

            "latitude":
                latitude,

            "longitude":
                longitude,

            "current": ",".join([

                "temperature_2m",

                "relative_humidity_2m",

                "apparent_temperature",

                "precipitation",

                "rain",

                "weather_code",

                "wind_speed_10m",

                "wind_direction_10m",

                "wind_gusts_10m"

            ]),

            "hourly": ",".join([

                "temperature_2m",

                "relative_humidity_2m",

                "precipitation_probability",

                "precipitation",

                "rain",

                "wind_speed_10m",

                "wind_gusts_10m",

                "et0_fao_evapotranspiration"

            ]),

            "daily": ",".join([

                "temperature_2m_max",

                "temperature_2m_min",

                "precipitation_sum",

                "rain_sum",

                "precipitation_probability_max",

                "wind_speed_10m_max",

                "weather_code"

            ]),

            "timezone":
                "auto",

            "forecast_days":
                7
        }


        response = requests.get(

            WEATHER_URL,

            params=params,

            timeout=20

        )


        response.raise_for_status()


        weather_data = (
            response.json()
        )


        current = (
            weather_data.get(
                "current",
                {}
            )
        )


        hourly = (
            weather_data.get(
                "hourly",
                {}
            )
        )


        daily = (
            weather_data.get(
                "daily",
                {}
            )
        )


        # ----------------------------------------------------
        # RAINFALL CALCULATIONS
        # ----------------------------------------------------

        precipitation_values = (
            hourly.get(
                "precipitation",
                []
            )
        )


        rain_values = (
            hourly.get(
                "rain",
                []
            )
        )


        probability_values = (
            hourly.get(
                "precipitation_probability",
                []
            )
        )


        next_24_precipitation = sum(
            precipitation_values[:24]
        )


        next_24_rain = sum(
            rain_values[:24]
        )


        next_24_probability = max(
            probability_values[:24],
            default=0
        )


        daily_precipitation = (
            daily.get(
                "precipitation_sum",
                []
            )
        )


        daily_rain = (
            daily.get(
                "rain_sum",
                []
            )
        )


        today_precipitation = (

            daily_precipitation[0]

            if daily_precipitation

            else 0

        )


        today_rain = (

            daily_rain[0]

            if daily_rain

            else 0

        )


        seven_day_precipitation = sum(
            daily_precipitation[:7]
        )


        seven_day_rain = sum(
            daily_rain[:7]
        )


        # ----------------------------------------------------
        # AGRICULTURAL ALERTS
        # ----------------------------------------------------

        alerts = generate_agriculture_alerts(

            current,

            hourly

        )


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "success":
                True,

            "data": {

                "location": {

                    "name":
                        location_name,

                    "state":
                        state,

                    "district":
                        district,

                    "country":
                        country,

                    "latitude":
                        latitude,

                    "longitude":
                        longitude

                },


                "current": {

                    "temperature":
                        current.get(
                            "temperature_2m"
                        ),

                    "humidity":
                        current.get(
                            "relative_humidity_2m"
                        ),

                    "apparent_temperature":
                        current.get(
                            "apparent_temperature"
                        ),

                    "precipitation":
                        current.get(
                            "precipitation"
                        ),

                    "rain":
                        current.get(
                            "rain"
                        ),

                    "wind_speed":
                        current.get(
                            "wind_speed_10m"
                        ),

                    "wind_gust":
                        current.get(
                            "wind_gusts_10m"
                        ),

                    "weather":
                        weather_description(
                            current.get(
                                "weather_code"
                            )
                        )

                },


                "rainfall": {

                    "current_rain":
                        current.get(
                            "rain",
                            0
                        ),

                    "next_24h_precipitation":
                        round(
                            next_24_precipitation,
                            2
                        ),

                    "next_24h_rain":
                        round(
                            next_24_rain,
                            2
                        ),

                    "next_24h_probability":
                        next_24_probability,

                    "today_precipitation":
                        round(
                            today_precipitation,
                            2
                        ),

                    "today_rain":
                        round(
                            today_rain,
                            2
                        ),

                    "seven_day_precipitation":
                        round(
                            seven_day_precipitation,
                            2
                        ),

                    "seven_day_rain":
                        round(
                            seven_day_rain,
                            2
                        )

                },


                "hourly":
                    hourly,


                "daily":
                    daily,


                "alerts":
                    alerts,


                "updated_at":
                    pd.Timestamp.now().isoformat()

            }

        })


    except requests.RequestException as e:

        print(
            "Weather API error:"
        )

        print(
            traceback.format_exc()
        )


        return jsonify({

            "success":
                False,

            "error":
                "Weather service is unavailable.",

            "details":
                str(e)

        }), 503


    except Exception as e:

        print(
            "Weather error:"
        )

        print(
            traceback.format_exc()
        )


        return jsonify({

            "success":
                False,

            "error":
                "Weather request failed.",

            "details":
                str(e)

        }), 500


# ============================================================
# CROP RECOMMENDATION
# ============================================================

@app.route(
    "/api/recommend-crop",
    methods=["POST"]
)
def recommend_crop():

    try:

        data = request.get_json()


        if not data:

            return jsonify({

                "success":
                    False,

                "error":
                    "No input data received."

            }), 400


        features = [

            "N",

            "P",

            "K",

            "temperature",

            "humidity",

            "ph",

            "rainfall"

        ]


        values = []


        for feature in features:

            if feature not in data:

                return jsonify({

                    "success":
                        False,

                    "error":
                        (
                            f"Missing required "
                            f"field: {feature}"
                        )

                }), 400


            try:

                value = float(
                    data[feature]
                )

            except (
                ValueError,
                TypeError
            ):

                return jsonify({

                    "success":
                        False,

                    "error":
                        (
                            f"{feature} "
                            f"must be numeric."
                        )

                }), 400


            values.append(
                value
            )


        X = pd.DataFrame(

            [values],

            columns=features

        )


        X_scaled = (
            crop_scaler.transform(X)
        )


        probabilities = (
            crop_model.predict(
                X_scaled,
                verbose=0
            )[0]
        )


        top_indices = np.argsort(
            probabilities
        )[::-1][:5]


        recommendations = []


        for index in top_indices:

            crop_name = (

                crop_label_encoder

                .inverse_transform(
                    [index]
                )[0]

            )


            confidence = (

                float(
                    probabilities[index]
                )

                * 100

            )


            recommendations.append({

                "crop":
                    str(crop_name),

                "confidence":
                    round(
                        confidence,
                        2
                    )

            })


        primary = (
            recommendations[0]
        )


        return jsonify({

            "success":
                True,

            "recommendation": {

                "crop":
                    primary["crop"],

                "confidence":
                    primary["confidence"]

            },

            "top_recommendations":
                recommendations

        })


    except Exception as e:

        print(
            traceback.format_exc()
        )


        return jsonify({

            "success":
                False,

            "error":
                "Crop recommendation failed.",

            "details":
                str(e)

        }), 500


# ============================================================
# CROP YIELD PREDICTION
# ============================================================

@app.route(
    "/api/predict-yield",
    methods=["POST"]
)
def predict_yield():

    try:

        data = request.get_json()


        if not data:

            return jsonify({

                "success":
                    False,

                "error":
                    "No input data received."

            }), 400


        required_fields = [

            "Crop",

            "Season",

            "State",

            "Area",

            "Production",

            "Annual_Rainfall",

            "Fertilizer",

            "Pesticide"

        ]


        for field in required_fields:

            if field not in data:

                return jsonify({

                    "success":
                        False,

                    "error":
                        (
                            f"Missing required "
                            f"field: {field}"
                        )

                }), 400


        input_data = pd.DataFrame([{

            "Crop":
                data["Crop"],

            "Season":
                data["Season"],

            "State":
                data["State"],

            "Area":
                float(data["Area"]),

            "Production":
                float(data["Production"]),

            "Annual_Rainfall":
                float(
                    data["Annual_Rainfall"]
                ),

            "Fertilizer":
                float(
                    data["Fertilizer"]
                ),

            "Pesticide":
                float(
                    data["Pesticide"]
                )

        }])


        encoded_data = (
            yield_encoder.transform(
                input_data
            )
        )


        encoded_data = (
            encoded_data.astype(
                np.float32
            )
        )


        predicted_log_yield = (

            yield_model.predict(
                encoded_data
            )

        )


        predicted_yield = np.expm1(
            predicted_log_yield
        )


        predicted_yield = np.maximum(
            predicted_yield,
            0
        )


        yield_value = float(
            predicted_yield[0]
        )


        return jsonify({

            "success":
                True,

            "prediction": {

                "yield":
                    round(
                        yield_value,
                        3
                    ),

                "unit":
                    "dataset yield unit",

                "area":
                    float(
                        data["Area"]
                    ),

                "crop":
                    data["Crop"],

                "state":
                    data["State"],

                "season":
                    data["Season"]

            }

        })


    except Exception as e:

        print(
            traceback.format_exc()
        )


        return jsonify({

            "success":
                False,

            "error":
                "Yield prediction failed.",

            "details":
                str(e)

        }), 500


# ============================================================
# CHATBOT
# ============================================================

def agri_mitra_chat(
    message,
    language="English"
):

    message_lower = (
        message.lower().strip()
    )


    if message_lower in [
        "hi",
        "hello",
        "hey"
    ]:

        if (
            language.lower()
            == "telugu"
        ):

            return (
                "నమస్కారం! నేను "
                "అగ్రిమిత్ర. వ్యవసాయం "
                "గురించి మీ ప్రశ్న అడగండి."
            )


        return (
            "Namaste! I am AgriMitra. "
            "Ask me anything about farming."
        )


    if "crop" in message_lower:

        return (
            "I can help with crop "
            "recommendation. Open the "
            "Crop Recommendation tool "
            "and enter your soil and "
            "weather values."
        )


    if "yield" in message_lower:

        return (
            "I can help estimate crop "
            "yield. Open the Crop Yield "
            "Prediction tool and enter "
            "your farm information."
        )


    return (
        "Your AgriMitra chatbot endpoint "
        "is connected. Replace this "
        "function with your Gradio/RAG "
        "chatbot inference when needed."
    )


@app.route(
    "/api/chat",
    methods=["POST"]
)
def chat():

    try:

        data = request.get_json()


        if not data:

            return jsonify({

                "success":
                    False,

                "error":
                    "No chat data received."

            }), 400


        message = str(
            data.get(
                "message",
                ""
            )
        ).strip()


        language = str(
            data.get(
                "language",
                "English"
            )
        )


        if not message:

            return jsonify({

                "success":
                    False,

                "error":
                    "Message cannot be empty."

            }), 400


        answer = agri_mitra_chat(

            message,

            language

        )


        return jsonify({

            "success":
                True,

            "message":
                answer,

            "language":
                language

        })


    except Exception as e:

        print(
            traceback.format_exc()
        )


        return jsonify({

            "success":
                False,

            "error":
                "Chatbot request failed.",

            "details":
                str(e)

        }), 500


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    load_models()


    print("\n")

    print("=" * 70)

    print(
        "AGRIMITRA AI SERVER"
    )

    print("=" * 70)

    print(
        "Open:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "Location API:"
        " /api/weather/locations"
    )

    print(
        "Weather API:"
        " /api/weather"
    )

    print("=" * 70)


    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )