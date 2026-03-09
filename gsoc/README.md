# Exercises 4–5

You can find the code for these exercises in their respective folders.  
I also wrote some tests in the `test` folder that you can run using `pytest`.

# hls4ml and SOFIE

In `main.py`, you can find a possible integration flow between SOFIE and `hls4ml`. The idea is to create a custom backend in `hls4ml` for SOFIE.

In the `sofie_backend` directory, you can find the source code of the new backend. In `main.py`, I register the backend and its writer.

The backend follows the default flow used by other HLS backends. In Python, given a model from a popular framework, you can parse the model and pass it to the SOFIE backend. You can then compile the model, which will:

- Write the RModel
- Generate the C++ code
- Compile a top function to run the session

Alternatively, you can extract the SOFIE session directly using the following static method:

```python
SofieBackend.get_sofie_session(hls_model)
```

This way, SOFIE could reuse the existing hls4ml tests and the optimization flows already implemented.
