from langchain_unstructured import UnstructuredLoader

loader = UnstructuredLoader(
    file_path='./dummy.pdf',
    strategy="hi_res",
    infer_table_structure=True,
    hi_res_model_name="yolox"
)

for doc in loader.load():
    pass
print('YoloX downloaded successfully!')
