import copy
import os
import torch
from PIL import Image
from ikomia import core, dataprocess, utils

from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferQwen3VlParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.model_name = "Qwen/Qwen3-VL-4B-Instruct"
        self.cuda = torch.cuda.is_available()
        self.prompt = 'Read all the text in the image.'
        self.system_prompt = 'You are a helpful assistant.'
        self.max_new_tokens = 256
        self.do_sample = False
        self.temperature = 0
        self.top_p = 1
        self.top_k = 5
        self.repetition_penalty = 1.0
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia Studio or API
        self.model_name = str(param_map["model_name"])
        self.cuda = utils.strtobool(param_map["cuda"])
        self.prompt = str(param_map["prompt"])
        self.system_prompt = str(param_map["system_prompt"])
        self.do_sample = utils.strtobool(param_map["do_sample"])
        self.max_new_tokens = int(param_map["max_new_tokens"])
        self.temperature = float(param_map["temperature"])
        self.top_p = float(param_map["top_p"])
        self.top_k = int(param_map["top_k"])
        self.repetition_penalty = float(param_map["repetition_penalty"])
        self.update = True


    def get_values(self):
        # Send parameters values to Ikomia Studio or API
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["model_name"] = str(self.model_name)
        param_map["prompt"] = str(self.prompt)
        param_map["system_prompt"] = str(self.system_prompt)
        param_map["max_new_tokens"] = str(self.max_new_tokens)
        param_map["do_sample"] = str(self.do_sample)
        param_map["temperature"] = str(self.temperature)
        param_map["top_p"] = str(self.top_p)
        param_map["top_k"] = str(self.top_k)
        param_map["repetition_penalty"] = str(self.repetition_penalty)
        param_map["cuda"] = str(self.cuda)
        return param_map



# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferQwen3Vl(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        self.add_output(dataprocess.DataDictIO())
        # Create parameters object
        if param is None:
            self.set_param_object(InferQwen3VlParam())
        else:
            self.set_param_object(copy.deepcopy(param))
        current_param = self.get_param_object()
        self.model_name = current_param.model_name
        self.model = None
        self.processor = None
        self.min_pixels = 256*28*28 
        self.max_pixels = 1280*28*28
        self.base_dir = os.path.dirname(os.path.realpath(__file__))
        self.model_folder = os.path.join(self.base_dir, "weights")
        self.device = torch.device("cpu")


    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def load_model(self):
        param = self.get_param_object()
        self.device = torch.device(
            "cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")
        # Prefer bfloat16 when using FP8 models; otherwise keep previous behavior
        if param.cuda and torch.cuda.is_available():
            if isinstance(param.model_name, str) and "FP8" in param.model_name.upper():
                torch_tensor_dtype = torch.bfloat16
            else:
                torch_tensor_dtype = torch.float16
        else:
            torch_tensor_dtype = torch.float32
        # Initialize model and processor
        self.model = AutoModelForImageTextToText.from_pretrained(
            param.model_name,
            torch_dtype=torch_tensor_dtype,
            device_map=self.device,
            cache_dir=self.model_folder
        )
        self.processor = AutoProcessor.from_pretrained(
            param.model_name,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            cache_dir=self.model_folder
        )

        param.update = False

    def init_long_process(self):
        self.load_model()
        super().init_long_process()

    def run(self):
        # Main function of your algorithm
        # Call begin_task_run() for initialization
        self.begin_task_run()
        # Get parameters
        param = self.get_param_object()

        # Get input image (np array):
        img_input = self.get_input(0)

        # Get image from input/output (numpy array):
        src_image = img_input.get_image()

        # transform image to PIL format
        src_image = Image.fromarray(src_image)

        # Set output
        output_dict = self.get_output(1)

        if param.update:
            self.load_model()       

        # Get parameters
        param = self.get_param_object()

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": param.system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": src_image},
                    {"type": "text", "text": param.prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        with torch.no_grad():
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=param.max_new_tokens,
                do_sample=param.do_sample,
                temperature=param.temperature,
                top_p=param.top_p,
                top_k=param.top_k,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print(output_text)

        output_dict.data = {
                            "reponse:": output_text,
                                }

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferQwen3VlFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_qwen3_vl"
        self.info.short_description = "Run vision-language model series based on Qwen3"
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/VLM"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "Qwen team"
        self.info.article = "Qwen3 Technical Report"
        self.info.journal = "arXiv:2505.09388"
        self.info.year = 2025
        self.info.license = "Apache 2.0"

        # Ikomia API compatibility
        self.info.min_ikomia_version = "0.13.0"

        # Python compatibility
        self.info.min_python_version = "3.11.0"

        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/abs/2502.13923"

        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_qwen3_vl"
        self.info.original_repository = "https://github.com/QwenLM/Qwen3-VL"

        # Keywords used for search
        self.info.keywords = "VLM,Qwen,Qwen3,VL,Vision-Language"

        # General type: INFER, TRAIN, DATASET or OTHER
        self.info.algo_type = core.AlgoType.INFER

        # Min hardware config
        self.info.hardware_config.min_cpu = 4
        self.info.hardware_config.min_ram = 16
        self.info.hardware_config.gpu_required = False
        self.info.hardware_config.min_vram = 6

    def create(self, param=None):
        # Create algorithm object
        return InferQwen3Vl(self.info.name, param)
