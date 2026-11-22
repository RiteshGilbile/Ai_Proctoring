from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import subprocess
import sys
import os

app = FastAPI(title="AI Exam Proctoring System")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "UI", "static")),
    name="static"
)

templates = Jinja2Templates(
    directory=os.path.join(BASE_DIR, "UI", "templates")
)


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


@app.post("/api/register", response_class=HTMLResponse)
def register_student(
    request: Request,
    student_name: str = Form(...)
):

    register_script = os.path.join(
        BASE_DIR,
        "register_student.py"
    )

    result = subprocess.run(
        [sys.executable, register_script, student_name],
        cwd=BASE_DIR
    )

    if result.returncode == 0:
        message = "Student registration completed successfully."
    else:
        message = "Student registration was cancelled or failed."

    return templates.TemplateResponse(
        request=request,
        name="register.html",
        context={
            "message": message
        }
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