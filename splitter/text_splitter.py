from langchain.text_splitter import MarkdownHeaderTextSplitter


def get_markdown_splitter():
    headers_to_split_on = [
        ("#", "Chapter"),
        ("##", "Section"),
        ("###", "Subsection"),
        ("####", "Sub-subsection")
    ]
    return MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
