# sprint 3
Medical Visual Question Answering

Group: Ruiling Zhang, Yuko Ishikawa, Zihao Shen

## Overall:
We duplicate and test an open-source python project, which is based on pytorch.

https://github.com/tbmoon/basic_vqa

Then we try to use a medical VQA dataset.

https://osf.io/2ku83/

## Dataset

VQA-RAD is a manufactured data set in which the questions and answers are radiographic images provided by clinicians. The pictures are all about radiology. There are 3,515 questions in 11 types. 

Clinical problems can be divided into four categories: modality problems, plane problems, organ system problems and abnormality problems. For the first three categories, QA uses a multiple choice (MC) style, with a fixed number of possible answers (36, 16, and 10, respectively). Therefore, QA tasks can be equivalently expressed as multipath classification problems with 36, 16, and 10 categories, respectively. This makes this data set much less difficult.

### 1) Modality
>Yes/No, WH and closed questions. Examples:  
>– was gi contrast given to the patient?  
>– what is the mr weighting in this image?  
>– what modality was used to take this image?  
>– is this a t1 weighted, t2 weighted, or flair image?  

### 2) Plane:
>WH questions.  
Examples:  
– what is the plane of this mri?  
– in what plane is this mammograph taken?  

### 3) Organ System:
>WH questions.  
Examples:  
– what organ system is shown in this x-ray?  
– what is the organ principally shown in this mri?  

### 4) Abnormality:
>Yes/No and WH questions.  
Examples:  
– does this image look normal?  
– are there abnormalities in this gastrointestinal image?  
– what is the primary abnormality in the image?  
– what is most alarming about this ultrasound?  


58% of QA in VQA-RAD are multiple choice questions, which means,  Most questions have a fixed number of candidate answers that can be sorted in a variety of ways. And the rest are open questions. 

## code

```
def build_medical_input():
    j = open('./VQA_RAD Dataset Public.json')
    info = json.load(j)
    dataset = [None]*len(info)
    for n_info, information in enumerate(info):
        image_name=information['image_name']
        image_path=os.path.join(RESIZE_PATH + '/', image_name)
        # print(image_path)
        question_id=information['qid']
        question_str=information['question']
        question_tokens=text_helper.tokenize(question_str)
        all_answers = [str(information['answer'])]
        iminfo = dict(
            image_name = image_name,
            image_path = image_path,
            question_id = question_id,
            question_str = question_str,
            question_tokens = question_tokens,
            all_answers = all_answers,
            valid_answers = all_answers)

        dataset[n_info] = iminfo
    print(dataset[0])
    data_array = np.array(dataset)
    length = len(data_array)
    train_arr = data_array
    valid_arr = data_array[int(length*0.9)+1:length]

    np.save(DATA_PATH+'/train.npy', train_arr)
    np.save(DATA_PATH+'/valid.npy', valid_arr)
```
![avatar](pic/2.png)

Now we only use the part of question and answer, other information has not been used yet.

## Future work:
![avatar](pic/1.png)

Medical VQA datasets are more challenging to construct than VQA datasets in the general field. For medical data, a piece of data is a patient's information. Medical images such as pathological images are highly domain specific, which can only be explained by well-educated medical professionals. 

Besides, to create a VQA dataset, a image dataset need to be collected first. Despite the ubiquity of images in the common domain, medical images are difficult to obtain because of privacy concerns.

For medical data, a piece of data is a patient's information. VQA-RAD is special, although it's small, it contants lots of information.

In next steps we will search for methods to make more effective use of the dataset information.