class MemoryPacketSink:
    def __init__(self): self.packets=[]
    def write_packet(self,packet): self.packets.append(packet)
