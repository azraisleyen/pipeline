class OnlineApiClient:
    def __init__(self, base_url='', token=''): self.base_url=base_url; self.token=token
    def fetch_frame(self): raise NotImplementedError('Competition API binding must provide fetch_frame')
    def submit_packet(self, packet): raise NotImplementedError('Competition API binding must provide submit_packet')
