import json, random, os

root = r'/raid/wsy/comperition/nqg-paragraph'
# root = r'F:/文本生成任务/competition-tianchi/nqg-paragraph/'
TRAIN_FILE_ORIGIN = root + '/dataset/round1_train_0907_origin.json'
TRAIN_FILE = root + '/dataset/round1_train_0907.json'
TEST_FILE = root + '/dataset/round1_test_0907.json'
DEV_FILE = root + '/dataset/round1_dev_0907.json'
RESULT_FILE=root+'/dataset/result.json'
dev_ratio = 0.1

train_src_file = os.path.join(root, "dataset/para-train.txt")
train_trg_file = os.path.join(root, "dataset/tgt-train.txt")
train_ans_file = os.path.join(root, "dataset/ans-train.txt")

dev_src_file = os.path.join(root, "dataset/para-dev.txt")
dev_trg_file = os.path.join(root, "dataset/tgt-dev.txt")
dev_ans_file = os.path.join(root, "dataset/ans-dev.txt")

test_src_file = os.path.join(root, "dataset/para-test.txt")
test_trg_file = os.path.join(root, "dataset/tgt-test.txt")
test_ans_file = os.path.join(root, "dataset/ans-test.txt")

generated_file=os.path.join(root, "result/pointer_maxout_ans/generated.txt")

def split_dev():
    with open(TRAIN_FILE_ORIGIN, 'r', encoding='utf-8') as train_file:
        train_data = json.load(train_file)
        picknumber = int(len(train_data) * dev_ratio)
        sample = random.sample(train_data, picknumber)
        json.dump(sample, open(DEV_FILE, 'w', encoding='utf-8'), ensure_ascii=False)
        for each in sample:
            train_data.remove(each)
        json.dump(train_data, open(TRAIN_FILE, 'w', encoding='utf-8'), ensure_ascii=False)


def process_file(data_file):
    with open(data_file, 'r', encoding="utf-8") as file:
        content = json.load(file)
        articles_all = []
        questions_all = []
        answers_all = []
        for each in content:
            article = ' '.join(each["text"].split("\n"))
            annotation = each["annotations"]
            for qa in annotation:
                articles_all.append(article)
                questions_all.append(" ".join(qa["Q"].split("\n")))
                answers_all.append(" ".join(qa["A"].split("\n")))
    return articles_all, questions_all, answers_all


def make_conll_format(sentences, questions, answers, src_file, tgt_file, ans_file):
    src_fw = open(src_file, "w", encoding='utf-8')
    trg_fw = open(tgt_file, "w", encoding='utf-8')
    ans_fw = open(ans_file, 'w', encoding='utf-8')
    assert len(sentences) == len(questions)
    assert len(sentences) == len(answers)
    for i in range(len(answers)):
        src_fw.write(sentences[i] + '\n')
        trg_fw.write(questions[i] + '\n')
        ans_fw.write(answers[i] + '\n')

    src_fw.close()
    trg_fw.close()
    ans_fw.close()


def trans_result(generated_file, answers_file):
    generated = open(generated_file, 'r', encoding='utf-8')
    answers = open(answers_file, 'r', encoding='utf-8')
    with open(TEST_FILE, 'r', encoding='utf-8') as test_file:
        count=0
        test_data = json.load(test_file)
        generated_data = generated.readlines()
        answers_data = answers.readlines()
        for each in test_data:
            for qa in each["annotations"]:
                ans=" ".join(qa["A"].split("\n"))+"\n"
                index=answers_data.index(ans)
                qa["Q"]=generated_data[index]
                count+=1
        print(count)
        json.dump(test_data,open(RESULT_FILE, 'w', encoding='utf-8'), ensure_ascii=False)



# sentence_all, question_all, answers_all = process_file(TRAIN_FILE)
# make_conll_format(sentence_all, question_all, answers_all, train_src_file, train_trg_file, train_ans_file)

# dev_sentence_all, dev_question_all, dev_answers_all = process_file(DEV_FILE)
# make_conll_format(dev_sentence_all, dev_question_all, dev_answers_all, dev_src_file, dev_trg_file, dev_ans_file)

# test_sentence_all, test_question_all, test_answers_all = process_file(TEST_FILE)
# make_conll_format(test_sentence_all, test_question_all, test_answers_all, test_src_file, test_trg_file, test_ans_file)
trans_result(generated_file,test_ans_file)