from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate algorithm object
        from infer_qwen3_vl.infer_qwen3_vl_process import InferQwen3VlFactory
        return InferQwen3VlFactory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from infer_qwen3_vl.infer_qwen3_vl_widget import InferQwen3VlWidgetFactory
        return InferQwen3VlWidgetFactory()
