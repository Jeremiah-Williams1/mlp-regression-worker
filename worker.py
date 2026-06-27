from base import run_worker
from predict import make_prediction


def predict_fn(input_data: dict) -> dict:
    return {"predicted_mpg": make_prediction(input_data)}


if __name__ == "__main__":
    run_worker(job_type="regression", predict_fn=predict_fn)
