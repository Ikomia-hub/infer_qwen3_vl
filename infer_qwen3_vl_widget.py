from torch.cuda import is_available
from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_qwen3_vl.infer_qwen3_vl_process import InferQwen3VlParam

# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the algorithm
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferQwen3VlWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferQwen3VlParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Model name
        self.edit_model = pyqtutils.append_edit(
            self.grid_layout, "Model name", self.parameters.model_name)
        
                # Prompt
        self.edit_prompt = pyqtutils.append_edit(self.grid_layout, "Prompt", self.parameters.prompt)
        self.edit_system_prompt = pyqtutils.append_edit(self.grid_layout, "System Prompt", self.parameters.system_prompt)

        # Cuda
        self.check_cuda = pyqtutils.append_check(
            self.grid_layout, "Cuda", self.parameters.cuda and is_available())

        # Max New Tokens
        self.spin_max_new_tokens = pyqtutils.append_spin(
            self.grid_layout, "Max New Tokens", self.parameters.max_new_tokens, min=1, max=10000)

        # Do Sample (checkbox)
        self.check_do_sample = pyqtutils.append_check(
            self.grid_layout, "Do Sample", self.parameters.do_sample)

        # Temperature
        self.spin_temperature = pyqtutils.append_double_spin(
            self.grid_layout, "Temperature", self.parameters.temperature, min=0.0, max=5.0, step=0.1, decimals=1)

        # Top P
        self.spin_top_p = pyqtutils.append_double_spin(
            self.grid_layout, "Top P", self.parameters.top_p, min=0.0, max=1.0, step=0.001, decimals=3)

        # Top K
        self.spin_top_k = pyqtutils.append_spin(
            self.grid_layout, "Top K", self.parameters.top_k, min=1, max=50)

        # Repetition Penalty
        self.spin_repetition_penalty = pyqtutils.append_double_spin(
            self.grid_layout, "Repetition Penalty", self.parameters.repetition_penalty, min=0.0, max=10.0, step=0.1, decimals=2)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Update parameters from widget values
        self.parameters.model_name = self.edit_model.text()
        self.parameters.prompt = self.edit_prompt.text()
        self.parameters.system_prompt = self.edit_system_prompt.text()
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.max_new_tokens = self.spin_max_new_tokens.value()
        self.parameters.do_sample = self.check_do_sample.isChecked()
        self.parameters.temperature = self.spin_temperature.value()
        self.parameters.top_p = self.spin_top_p.value()
        self.parameters.top_k = self.spin_top_k.value()
        self.parameters.repetition_penalty = self.spin_repetition_penalty.value()
        self.parameters.update = True

        # Send signal to launch the process
        self.emit_apply(self.parameters)

# --------------------
# - Factory class to build algorithm widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferQwen3VlWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "infer_qwen3_vl"

    def create(self, param):
        # Create widget object
        return InferQwen3VlWidget(param, None)
