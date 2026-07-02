from base import run_worker
from predict import make_prediction


def predict_fn(input_data: dict) -> dict:
    result = make_prediction(input_data)
    return result


if __name__ == "__main__":
    run_worker(job_type="regression", predict_fn=predict_fn)
