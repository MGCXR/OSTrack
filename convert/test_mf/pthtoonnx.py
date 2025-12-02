import os
import sys
import torch
import onnx
import argparse
import importlib
from lib.models.test.test import build_test
from torch.serialization import add_safe_globals
import onnxsim 



class AverageMeter:
    pass

add_safe_globals([AverageMeter])

if __name__ == "__main__":

    # print(sys.path)
    # Load the PyTorch model
    model_params = torch.load('./convert/test_mf/models/Test_ep0300.pth.tar', map_location='cpu',weights_only=False)['net']
    # print(model_params.keys())
    


    config_module = importlib.import_module('lib.config.test.config')
    cfg = config_module.cfg
    config_module.update_config_from_file('./experiments/test/vitb_256_mae_ce_32x4_ep300_hybrid_hw_mf.yaml')
    model = build_test(cfg,training=False)
    # print(model)
    model.load_state_dict(model_params, strict=True)
    model.eval()
    print("Model loaded successfully.")
    # # Create a dummy input tensor with the appropriate shape
    template = torch.randn(1, 3, 128, 128)
    search = torch.randn(1, 3, 256, 256)

    # Export the model to ONNX format
    torch.onnx.export(model, (template, search, template, search), './convert/test_mf/models/Test_ep0300.onnx', 
                      
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['template', 'search', 'template_event', 'search_event'],
                      output_names=['score_map', 'size_map', 'offset_map'],
                    #   dynamic_axes={'template': {0: 'batch_size'},
                    #                 'search': {0: 'batch_size'}, 
                    #                 'template_event': {0: 'batch_size'},
                    #                 'search_event': {0: 'batch_size'}
                    #                 },
                    dynamic_axes=None,
                    #   dynamo=True,
                      export_modules_as_functions=False
                      )
    print("Model has been exported to ONNX format.")
    
    onnx_model = onnx.load('./convert/test_mf/models/Test_ep0300.onnx')
    onnx_model_simplified, check = onnxsim.simplify(onnx_model)
    onnx.save(onnx_model_simplified, './convert/test_mf/models/output.onnx')
    print("ONNX model has been simplified and saved.",'model check:',check)
    
    torch.save(model,'./convert/test_mf/models/Test_ep0300.pth')
    print("Model has been saved as a standard PyTorch model.")

    print("All done.")