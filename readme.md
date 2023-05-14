## **Python code Runmning Instructions :**    
 - Our code contains multiple files such as : 
   - ***Model_mnist.py*** - it is basically for applying neural network to the dataset mnist and getting the model trained and ready for the operator reordering.
   - ***neural_conv_render.py*** - It is basically for the loading the preloaded mnist model into the onnx parser.
   - ***onnx_operator_reordering.py***  - It is file that contains onnx parser code , it contains multiple libraries like tensorrt and pycuda whose specific versions are needed to be downloaded in order to run the code.
   **TensorRT Version <=7.0.0** must be used to run this succesfully.
 - Libraries to be installed to run these code files.
    - numpy
    - pandas
    - matplotlib 
    - tensorrt 7.0.1
    - pycuda
    - tensortflow
    - keras
    - Mentioned DL lib and frameworks.
- Files with the results saved are  : 
    - **Functional_variance.py** :  It contains all the models ran already on some epoch values so no need to run it graphs can be visible via either jupyter notebook or colab notebook. 
- Files needed to run : 
    - **Mem_optimizer_grh.py** : This file is needed to be run to get the new graphs for variable epoch values it will take considerable amount of time to run.

**Files must be saved in the same directory in order to use the function imports of each other.**

**Two New model files are created on running model_mnist.py and onnx_operator_reordering.py they need special version of the libraries and some attributes are deprecated in the new_versions**