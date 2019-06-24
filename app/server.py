import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://drive.google.com/uc?export=download&id=1rV6kp1EO0Nf1HFVf3H0JpP7nmGPxXxHN'
export_file_name = 'model_5150.pkl'

classes = ['consumo', 'veneno']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)
class F1_Score:
    def __init__(self,thresh:float):
        self.thresh = thresh
        
    def __call__(self,inp,targ):
        targ = [[0,1]] * len(inp)
        targ = torch.tensor(targ).cuda()
        return fai.metrics.fbeta(inp, targ, thresh=self.thresh, beta=5)
    
    def __repr__(self):
        return f"F1({self.thresh})"
    
    @property
    def __name__(self):
        return self.__repr__()

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    data_bunch = ImageDataBunch.single_from_classes(path, classes,
        ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = cnn_learner(data_bunch, models.resnet34, pretrained=False)
    metrics = [F1_Score(t) for t in [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]]
    learner.metrics= metrics
    learn.load(model_file_name)
    return learn


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})





if __name__ == '__main__':
    
    class F1_Score:
        def __init__(self,thresh:float):
            self.thresh = thresh

        def __call__(self,inp,targ):
            targ = [[0,1]] * len(inp)
            targ = torch.tensor(targ).cuda()
            return fai.metrics.fbeta(inp, targ, thresh=self.thresh, beta=5)

        def __repr__(self):
            return f"F1({self.thresh})"

        @property
        def __name__(self):
            return self.__repr__()
        
        
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
    
