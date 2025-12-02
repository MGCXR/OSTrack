# 导入灵汐库
import lyngor as lyn

def load_classname(filename = "data/resnet/imagenet_class.txt"):
    import csv
    aa = {}
    with open(filename, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            xx = row[0].split('\t')
            aa.update({int(xx[0]): "%s-%s"%(xx[1], xx[2])})
    return aa


def convert_model(model_file, inputs_dict=None, target="llvm", path='./tmp_net/', model_type='Tensorflow', outputs=None, device=0, build_mode="abc_only", profiler=False, serialize=False):

    #1. 创建一个待加速的计算图（或对训练好的模型进行转换得到计算图）
    model = lyn.DLModel()
    model.load(model_file, model_type=model_type, inputs_dict=inputs_dict, outputs=outputs,in_type='float32', out_type='float32')

    #2. 创建一个Builder来编译计算图，并保存
    offline_builder = lyn.Builder(target=target)
    out_path = offline_builder.build(model.graph, model.params, out_path=path,serialize=serialize, build_mode=build_mode, profiler=profiler)
    print("模型已保存至：", out_path)
    #3. 直接Load即可得到runtime引擎
    # r_engine = lyn.load(path=out_path + "/Net_0/", device=device)
    r_engine = lyn.load(path=out_path, device=device)
    return r_engine
