from lib.train.dataset.visevent import VisEvent
import torch
if __name__ == '__main__':
    # dataset = VisEvent()

    # print(f'First sequence: {dataset.sequence_list[0]}')
    # # print(f'Class list: {dataset._get_class_list()}')
    # print(f'Dataset size: {len(dataset)}')
    # # print(f'sequence_info: {dataset.get_sequence_info(0)}')
    # print(f'get_frames: {dataset.get_frames(0, list(range(0, 10)))}')
    checkpoint = torch.load('./temp/OSTrack_ep0300.pth.tar', map_location="cpu",weights_only=False)
    print(checkpoint.keys())
    checkpoint['epoch'] = 1
    torch.save(checkpoint, './temp/OSTrack_ep0001.pth.tar')