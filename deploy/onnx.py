import torch
from models.FLENet import FLENet_T0



def pth_to_onnx ():
    model = FLENet_T0(num_classes=8105)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_weight_path = "deploy/weights/FLENet_T0.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device),strict=False)
    model.eval()
    batch_size=10
    x = torch.randn(batch_size, 3, 128, 128, requires_grad=True)
    y = model(x)

    # Export the model
    export_onnx_file = "FLENet_T0.onnx" 
    torch.onnx.export(model,                     # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    export_onnx_file,     # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                    'output' : {0 : 'batch_size'}})


pth_to_onnx ()