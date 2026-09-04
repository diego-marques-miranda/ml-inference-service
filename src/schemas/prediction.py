import pydantic as pyd 

class FeaturesSchema(pyd.BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float

class PredictionSchema(pyd.BaseModel):
    prediction: float