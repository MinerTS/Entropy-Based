import math
import random

with open('glass.txt', 'r') as file:
    lines = file.readlines()
# 自定義column name
column_names = ['Id number', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe','Type of glass']
Attributes = [0,1,2,3,4,5,6,7,8]
# 初始化字典
raw_data = []
Ground_tru = []

for line in lines:
    row = line.strip().split(',')
    Ground_tru.append(row.pop(10))
    if len(row) == len(Attributes)+1:
            instance = dict(zip(Attributes, row[1:]))
            raw_data.append(instance)

def probability(data) -> float:   #計算類別資料發生率公式
    total_samples = len(data)
    class_counts = {}
   
    for value in data:
        if value[1] in class_counts:
            class_counts[str(value[1])] += 1
        else:
            class_counts[str(value[1])] = 1

    prob = {key: counts / total_samples for key, counts in class_counts.items()}

    return prob

def H_entropy(data):   # 計算Entropye公式_H(X),H(Y)
    probs = probability(data)
    probs_values = [prob[1] for prob in probs.items()] # 抽出字典value到list中，[0]是key [1]是value


    entropy_value = -sum(p * math.log2(p) for p in probs_values)
    return entropy_value

def EntropyBased(Sorted_value, Splitting_point: list , min_value = 0, max_value = 0):
    currentBestGain = 0
    currentBestPoint = 0
    criterion = 0
    values_P = []

    if len(Splitting_point) == 0:
        values_P = sorted(Sorted_value)
        max_value = values_P[-1][0] + 1  # max
        min_value = values_P[0][0]  # min
    else:
        for sub_idx in Sorted_value:
            if min_value <= sub_idx[0] < max_value:
                values_P.append(sub_idx)
  
    Entropy_P = H_entropy(values_P)

    candidates_list = set()
    for i in range(len(values_P) - 1):  # construct the candidate list
        candidates_list.add(round((values_P[i][0] + values_P[i + 1][0]) / 2, 8))
    candidates_list = sorted(candidates_list)

    currentS1 = []
    currentS2 = []

    for point in candidates_list:
        S1 = []
        S2 = []
        for i in range(len(values_P)):
            if values_P[i][0] < point:
                S1.append(values_P[i])
            else:
                S2.append(values_P[i])
        X = (len(S1)/len(values_P)) * H_entropy(S1)
        Y = (len(S2)/len(values_P)) * H_entropy(S2)
        currentGain = Entropy_P - (X + Y)
        if currentGain > currentBestGain:
            currentBestGain = currentGain
            currentBestPoint = point
            currentS1 = S1
            currentS2 = S2
    S = len(values_P)   # 計算criterion是否符合MPL priciple，符合再下切
    currentX = H_entropy(currentS1)
    currentY = H_entropy(currentS2)
    k = len(set(values_P[j][1] for j in range(len(values_P))))
    k1 = len(set(currentS1[m][1] for m in range(len(currentS1))))
    k2 = len(set(currentS2[n][1] for n in range(len(currentS2))))
    criterion = math.log2(S - 1) / S + (math.log2(3 ** k - 2) - (k * Entropy_P - k1 * currentX - k2 * currentY)) / S
    if currentBestGain - criterion > 0:
        Splitting_value = currentBestPoint
        Splitting_point.append(Splitting_value)
        EntropyBased(Sorted_value, Splitting_point = Splitting_point, min_value = min_value, max_value = Splitting_value)
        EntropyBased(Sorted_value, Splitting_point = Splitting_point, min_value = Splitting_value, max_value = max_value)


discretized_row = []
discr_ent_data = []  #Entrop_Based後的資料，List包含Dict裡面的value為str

# 循環處理每個屬性
for class_selected in Attributes:
    attrTodis = [float(row[class_selected]) for row in raw_data]
    valuesTodis = list(zip(attrTodis, Ground_tru))

    Splitting_list = []
    EntropyBased(valuesTodis, Splitting_point = Splitting_list)
    Splitting_list.append(min(attrTodis))
    Splitting_list.append(max(attrTodis))
    Splitting_list.sort()

    discr_ent_str = []  # 開使做value mapping
    if len(Splitting_list) > 2:
        for value in attrTodis:
            for internvl in range(len(Splitting_list) - 1):
                if Splitting_list[internvl] <= value < Splitting_list[internvl + 1]:
                    discr_ent_str.append(str(internvl + 1))
            if value == max(attrTodis):
                discr_ent_str.append(str(len(Splitting_list) - 1))

    else:
        for i in range(len(attrTodis)):
            discr_ent_str.append(str(1))
    discretized_row.append(discr_ent_str)



    class_name = column_names[Attributes[class_selected] + 1]  # 屬性名稱mapping
    # print(f"Splitting Points(include Max. and Min.) for {class_name}:{Splitting_list}")

for values in zip(*discretized_row):
    discretized_dict = {key: value for key, value in zip(Attributes, values)}
    discr_ent_data.append(discretized_dict)

def laplace(target_class, attr, value, feature_idx):  
    laplace = (attribute_count[target_class][attr][value] + 1) / (class_count[target_class] + AttrProbValueNum[feature_idx])
    return laplace

#Naive Bayes classifier
def classifier(run_training_set, run_testing_set):
    
    for j in range(len(run_training_set[0])):   # instances idx in each fold
        for Ci in range(ClassProbValue + 1):   # Class可能值
            if run_training_set[len(run_training_set) - 1][j] == str(Ci):
                class_count[Ci] += 1
            for Ai in range(len(run_training_set) - 1):   # attribute idx
                for valueAi in range(1, MaxAttrProbValue + 1):
                    if run_training_set[len(run_training_set) - 1][j] == str(Ci) and int(run_training_set[Ai][j]) == valueAi:
                        attribute_count[Ci][Ai][valueAi] += 1
    
    
    Acc_count = 0

    for instance in range(len(run_testing_set[0])):
        predic = 0        
        best_likehood = 0

        for target_class in range(1,len(class_count)):
            likehood = class_count[target_class] / (len(run_training_set[0]))

            for attr in range(len(run_testing_set) - 1):
                value = int(run_testing_set[attr][instance])
                likehood *= laplace(target_class, attr, value, features_to_try[attr])

            if likehood > best_likehood:
                best_likehood = likehood
                predic = target_class

        if run_testing_set[len(run_testing_set) - 1][instance] == str(predic):
            Acc_count += 1

    return Acc_count

# 切割出5-fold
def fivefolds(discretized_row):
    row_array = discretized_row
    row_array.append(Ground_tru)
    reverse_arry = []
    for i in range(len(Ground_tru)):
        length = []
        for j in range(len(row_array)):
            length.append(row_array[j][i])
        reverse_arry.append(length)
    post_row = reverse_arry
    random.shuffle(post_row)

    #  設定每個fold的instances#
    fold_freq = len(Ground_tru) // 5
    fold_sizes = [fold_freq] * 5

    # 於平均分配前四個
    remainder = len(Ground_tru) % 5
    for i in range(remainder):
        fold_sizes[i] += 1

    fold_point = [0]
    for size in fold_sizes:
        next_point = fold_point[-1] + size
        fold_point.append(next_point)

    # 開始分割
    folds_data = []
    for i in range(5):
        one_fold = []
        for j in range(len(Ground_tru)):
            if fold_point[i] <= j < fold_point[i + 1]:
                one_fold.append(post_row[j])
        folds_data.append(one_fold)
    return folds_data


# 先選attribute，再5-folds cross vaildation
SNB_ordering_features = []

while True:
    feature_to_add = None
    current_best_acc = 0.0

    for feature_index in range(len(Attributes)):

        if feature_index not in SNB_ordering_features:
            # 複製已選Attributes列表，並加入候選Attributes
            features_to_try = SNB_ordering_features.copy()
            features_to_try.append(Attributes[feature_index])

            # 提取候選Attributes
            candidate_features = []
            for feature in features_to_try:
                candidate_features.append(discretized_row[feature])
            folds_data = fivefolds(candidate_features)

            # 先分出training/testing set，再計算出sample mean acc
            data_set = folds_data
            folds_acc_sum = 0
            for i in range(len(folds_data)):
                testing_set = []
                training_set = []
                testing_set.extend(data_set[i])
                for j in range(len(folds_data)):
                    if j != i:
                        training_set.extend(data_set[j])

                run_testing_set = [] # 整理testing_set
                for i in range(len(testing_set[0])):
                    app_testing_data = []
                    for j in range(len(testing_set)):
                        app_testing_data.append(testing_set[j][i])
                    run_testing_set.append(app_testing_data)

                run_training_set = [] # 整理training_set
                for i in range(len(training_set[0])):
                    app_training_data = []
                    for j in range(len(training_set)):
                        app_training_data.append(training_set[j][i])
                    run_training_set.append(app_training_data)

                # 計算先驗機率(Train)，把count結果紀錄在矩陣中，另外計算每個attributes的可能值 for Laplace
                MaxAttrProbValue = 0
                ClassProbValue = max(int(item) for item in Ground_tru)
                AttrProbValueNum = []   # call out by "feature_index"
                for i in Attributes:
                    x = [int(item) for item in discretized_row[i]]
                    ProbValueNum = max(x)
                    AttrProbValueNum.append(ProbValueNum)
                    if ProbValueNum > MaxAttrProbValue:
                        MaxAttrProbValue = ProbValueNum
                attribute_count = [[[0 for _ in range(MaxAttrProbValue + 1)] 
                                    for _ in range(len(Attributes))] 
                                    for _ in range(ClassProbValue + 1)]
                class_count = [0 for _ in range(ClassProbValue + 1)]

                acc_count = classifier(run_training_set, run_testing_set)
                fold_acc = acc_count / len(run_testing_set[0])
                folds_acc_sum += fold_acc
                # print(f"Correct prediction: {acc_count}, Fold size: {len(run_testing_set[0])}, Accuracy: {fold_acc:.4f}")
            current_acc_mean = folds_acc_sum / len(folds_data)
            if current_acc_mean > current_best_acc:
                current_best_acc = current_acc_mean
                feature_to_add = feature_index
    SNB_ordering_features.append(feature_to_add)
    if len(SNB_ordering_features) == len(Attributes):
        break
print("end")
    
