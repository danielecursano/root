import ROOT

def load_sofie_session(model_path):
        # Extract TMVA Session to run inference with the generated model. 
        model_name = model_path.split("/")[-1]
        ROOT.gInterpreter.Declare(f'#include "{model_path}.hxx"')
        sofie_project = getattr(ROOT, f"TMVA_SOFIE_{model_name}", None)
        if not sofie_project:
            raise RuntimeError(f"SOFIE namespace TMVA_SOFIE_{model_name} not found.")
        session = sofie_project.Session(model_path+".dat")
        return session
