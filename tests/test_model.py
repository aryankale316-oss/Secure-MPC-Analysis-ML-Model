from src.model.model import create_model

def test_model():

    model = create_model()

    assert model is not None

    print("Model test passed")
