from lib.train.dataset.visevent import VisEvent

if __name__ == '__main__':
    dataset = VisEvent()

    print(f'First sequence: {dataset.sequence_list[0]}')
    # print(f'Class list: {dataset._get_class_list()}')
    print(f'Dataset size: {len(dataset)}')
    # print(f'sequence_info: {dataset.get_sequence_info(0)}')
    print(f'get_frames: {dataset.get_frames(0, list(range(0, 10)))}')