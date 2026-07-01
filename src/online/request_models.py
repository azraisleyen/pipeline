from dataclasses import dataclass
@dataclass
class OnlineFrameRequest: frame_id:int; frame:any; user:str=''
