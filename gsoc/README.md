# Exercises 4–5

You can find the code for these exercises in their respective folders.  
I also wrote some tests in the `test` folder that you can run using `pytest`.

## hls4ml and SOFIE Integration

### Overview

In `main.py`, you can find a possible integration flow between SOFIE and `hls4ml`. The idea is to create a custom backend in `hls4ml` for SOFIE.

In the `sofie_backend` directory, you can find the source code of the new backend. In `main.py`, I register the backend and its writer.

### Workflow

The backend follows the default flow used by other HLS backends:

1. **Parse** - Load a model from a popular framework (Keras, PyTorch, etc.)
2. **Generate** - Pass to the SOFIE backend, which will:
   - Write the RModel
   - Generate the C++ code
   - Compile a top function to run the session
3. **Use** - Deploy in three ways:

#### Option A: Use ModelGraph predict (Python)
```python
hls_model.predict(x)
```

#### Option B: Extract SOFIE Session (Python)
```python
load_sofie_session(header_path)
```

#### Option C: Include Generated Code (C++)
Include the generated C++ code from `projectdir/projectname.hxx` in your C++ project.

By integrating with `hls4ml`, SOFIE can:
- Reuse existing hls4ml tests
- Leverage optimization flows already implemented
- Maintain compatibility with popular ML frameworks

## Issues

### Duplicate Session Names Load Wrong .dat File
**Problem:** When loading two sessions one after the other with the same name, SOFIE loads the correct header but the wrong .dat file.

**Solution:** Ensure every project has a unique name.

## Future Developments

- **Custom Layer Attributes** - hls4ml backends can override layer classes by adding custom attributes. Currently, SOFIE backend keeps the original classes from `hls4ml.model.layers` without modification. In the future, SOFIE may need to register additional attributes.
- **New Operators** - Implement additional operators as needed.
- **Model Architectures** - Support different types of model architectures beyond current scope.
- **C++ Testbenches** - Develop comprehensive input testbenches for pure C++ inference.