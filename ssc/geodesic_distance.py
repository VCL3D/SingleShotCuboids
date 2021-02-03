import torch

__all__ = [
    "_geodesic_distance",
    "GeodesicDistance",
]

def _geodesic_distance(
    long1:                  torch.Tensor,
    lat1:                   torch.Tensor,
    long2:                  torch.Tensor,
    lat2:                   torch.Tensor,
    epsilon:                float=1e-24,
):
    return 2.0 * torch.asin(
        torch.sqrt(        
            (torch.sin((lat1 - lat2) / 2.0)) ** 2.0 
            + torch.cos(lat1) * torch.cos(lat2) * (
                torch.sin((long1 - long2) / 2.0)
            ) ** 2.0 + epsilon
        )
    )

class GeodesicDistance(torch.nn.Module):
    def __init__(self,
        epsilon: float=1e-12,
    ):
        super(GeodesicDistance, self).__init__()
        self.epsilon = epsilon

    def forward(self, 
        gt:                 torch.Tensor,
        pred:               torch.Tensor,
    ) -> torch.Tensor:
        long1 = gt[:, :, 0]
        lat1 = gt[:, :, 1]
        long2 = pred[:, :, 0]
        lat2 = pred[:, :, 1]
        d = _geodesic_distance(long1, lat1, long2, lat2, self.epsilon)
        return d

if __name__ == "__main__":
    import cv2
    import numpy as np
    width = 1024
    height = width // 2
    img = np.zeros((height, width, 3))
    def callback(event, x, y, flags, param):
        global start_mouseX, start_mouseY, start_phi, start_theta
        global stop_mouseX, stop_mouseY, stop_phi, stop_theta
        global width, height        
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x,y), 3, (0, 0, 255), -1)
            start_mouseX, start_mouseY = x, y
            start_phi = (x / width) * (np.pi * 2) - np.pi
            start_theta = (y / height) * np.pi - (0.5 * np.pi)
        if event == cv2.EVENT_RBUTTONDOWN:
            cv2.circle(img, (x,y), 3, (255, 0, 0), -1)
            stop_mouseX, stop_mouseY = x, y
            stop_phi = (x / width) * (np.pi * 2) - np.pi
            stop_theta = (y / height) * np.pi - (0.5 * np.pi)
            dist = GeodesicDistance()
            start = torch.Tensor([[[start_phi, start_theta]]])
            stop = torch.Tensor([[[stop_phi, stop_theta]]])
            error = dist(start, stop)
            print(error)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', callback)
    while(1):
        cv2.imshow('image', img)
        k = cv2.waitKey(20) & 0xFF
        if k == 32:
            img.fill(0)            
        if k == 27:
            break