from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from fastapi.templating import Jinja2Templates

from src.pipelines.predict_pipeline import CustomData, PredictPipeline

app = FastAPI()

templates = Jinja2Templates(directory='templates')

@app.get('/', response_class=HTMLResponse)
async def hello_world(request: Request):
    return templates.TemplateResponse(
        request=request,
        name='index.html'
    )

@app.api_route('/predict', methods=['POST', 'GET'], response_class=HTMLResponse)
async def predict_data(request: Request):
    if request.method == 'POST':
        async with request.form() as form:
            gender = form.get('gender')
            race_ethnicity = form['race_ethnicity']
            parental_level_of_education = form['parental_level_of_education']
            lunch = form['lunch']
            test_preparation_course = form['test_preparation_course']
            reading_score = str(form.get('reading_score'))
            writing_score = str(form.get('writing_score'))

        dataset = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=int(reading_score),
            writing_score=int(writing_score)
        )
        dataframe = dataset.create_dataframe()
        print(dataframe)
        prediction = PredictPipeline()

        results = prediction.predict(dataframe=dataframe)

        return templates.TemplateResponse(
            request=request,
            name='home.html',
            context={
                'results': results[0]
            }
        )
    elif request.method == 'GET':
        return templates.TemplateResponse(
            request=request,
            name='home.html',
            context={
                'results': 0
            }
        )