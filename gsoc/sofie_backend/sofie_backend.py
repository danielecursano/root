import ROOT
import numpy as np
import subprocess

from hls4ml.backends import Backend
from hls4ml.model.flow import register_flow
from hls4ml.model.types import StandardFloatPrecisionType
from hls4ml.model.optimizer import model_optimizer
from hls4ml.writer import get_writer

class SofieBackend(Backend):
    def __init__(self):
        super().__init__('Sofie')
        initializers = self._get_layer_initializers()
        self._default_flow = register_flow('init_layers', initializers, requires=[], backend=self.name)
        self._writer_flow = register_flow('write', ['make_stamp', 'sofie:write_code'], requires=[self._default_flow], backend=self.name)
        self.writer = get_writer(self.name)
        
    def create_initial_config(self, **kwargs):
        return dict()
        
    def get_default_flow(self):
        return self._default_flow
            
    def get_writer_flow(self):
        return self._writer_flow
        
    def create_layer_class(self, layer_class):
        # No additional attributes for layer classes
        return layer_class
        
    def write(self, model):
        model.apply_flow(self.get_writer_flow())
        
    def compile(self, model):
        # Inherited from FPGABackend. ModelGraph still needs a .so to extract the top function
        lib_name = None
        ret_val = subprocess.run(
            ['./build_lib.sh'],
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=model.config.get_output_dir(),
        )
        if ret_val.returncode != 0:
            print(ret_val.stdout)
            raise Exception(f'Failed to compile project "{model.config.get_project_name()}"')
        lib_name = '{}/firmware/{}-{}.so'.format(
            model.config.get_output_dir(), model.config.get_project_name(), model.config.get_config_value('Stamp')
        )

        return lib_name
    
    @classmethod    
    def convert_precision_string(cls, precision):
        # SOFIE RModel supports only float precision at the moment
        return StandardFloatPrecisionType(width=32, exponent=8, use_cpp_type=True)
    
    @staticmethod
    def get_sofie_session(model):
        header_path = model.config.get_output_dir() + "/" + model.config.get_project_name()
        ROOT.gInterpreter.Declare(f'#include "{header_path}.hxx"')
        sofie_project = getattr(ROOT, f"TMVA_SOFIE_{model.config.get_project_name()}", None)
        if not sofie_project:
            raise RuntimeError(f"SOFIE namespace TMVA_SOFIE_{model.config.get_project_name()} not found.")
        session = sofie_project.Session(header_path+".dat")
        return session
        
    @model_optimizer()
    def write_code(self, model):
        self.writer.write(model)
        return True
        
        
        
