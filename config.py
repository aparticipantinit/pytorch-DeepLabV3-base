from yacs.config import CfgNode as CN
from utils.default import get_cfg_defaults

class args_c():
    def __init__(self,cfg_path=".\\my_model.yaml"):
        self.cfg=cfg_path
        self.config = get_cfg_defaults()
        self.model = None
        self.optimizer = None
        self.parse()
    def parse(self):
        self.config.defrost()  # 解冻参数，可以修改
        self.config.merge_from_file(self.cfg)
        self.config.freeze()
    def getConfig(self):
        return self.config.clone()

if __name__ == "__main__":
    a=args_c()
    print(a)
