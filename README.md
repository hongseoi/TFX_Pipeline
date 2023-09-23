# Sentiment analysis model within TFX_Pipeline
RNN 모델의 Text Data 감성분석 Pipeline 제작 Project

TFX를 활용하여 텍스트데이터 [감성분석](https://paperswithcode.com/task/sentiment-analysis)을 수행하는 RNN 모델의 파이프라인을 구축하고 streamlit을 활용해 웹서비스화 해볼 것이다


## To-do
박찬성님의 TFX Pipeline todo를 참고하였다.
- [ ] Notebook to prepare input dataset in `TFRecord` format
- [ ] Upload the input dataset into the GCS bucket
- [ ] Implement and include RNN model in the pipeline
- [ ] Implement Streamlit app template
- [ ] Make a complete TFX pipeline with `ExampleGen`, `SchemaGen`, `Resolver`, `Trainer`, `Evaluator`, and `Pusher` components
- [ ] Add necessary configurations to the [configs.py](https://github.com/deep-diver/semantic-segmentation-ml-pipeline/blob/main/training_pipeline/pipeline/configs.py)
- [ ] Add `HFPusher` component to the TFX pipeline
- [ ] Replace `SchemaGen` with `ImportSchemaGen` for better TFRecords parsing capability
- [ ] (Optional) Integrate `Dataflow` in `ImportExampleGen` to handle a large amount of dataset. This feature is included in the code as a reference, but it is not used after we switched the Sidewalk to PETS dataset.

## References
- [pipeline](https://github.com/deep-diver/semantic-segmentation-ml-pipeline/tree/main)
- [Model](https://github.com/hongseoi/aiffel-5-research/tree/master/EXPLORATION/EXPLORATION08)