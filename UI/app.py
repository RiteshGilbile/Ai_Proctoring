from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="AI Exam Proctoring System")

app.mount("/static", StaticFiles(directory="UI/static"), name="static")

templates = Jinja2Templates(directory="UI/templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.get("/register", response_class=HTMLResponse)
def register(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="register.html"
    )


@app.get("/exam", response_class=HTMLResponse)
def exam(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="exam.html"
    )


@app.get("/result", response_class=HTMLResponse)
def result(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="result.html"
    )


@app.get("/api/status")
def status():
    return {
        "status": "running",
        "message": "AI Exam Proctoring System is running"
    }