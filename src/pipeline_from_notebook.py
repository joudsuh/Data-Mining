from google.colab import drive
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import gradio as gr

def main():
    drive.mount('/content/drive')

    folder_path = "/content/drive/MyDrive/dm project"

    print(os.listdir(folder_path))





    def clean_numbers_and_symbols(text):
        text = re.sub(r"^\s*\d+\.\s*", "", text)
        text = re.sub(r"[-/]", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


    def remove_empty_lines(lines):
        return [line for line in lines if line.strip()]

    folder_path = "/content/drive/MyDrive/dm project"
    cleaned_folder_path = "/content/drive/MyDrive/dm_project_cleaned"
    os.makedirs(cleaned_folder_path, exist_ok=True)

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)

            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()

            cleaned_lines = [clean_numbers_and_symbols(line.strip()) for line in lines]


            cleaned_lines = remove_empty_lines(cleaned_lines)

            cleaned_file_path = os.path.join(cleaned_folder_path, file_name)
            with open(cleaned_file_path, "w", encoding="utf-8") as cleaned_file:
                cleaned_file.write("\n".join(cleaned_lines))

    print("All files have been cleaned and saved in:", cleaned_folder_path)




    # Lowercase Conversion
    for filename in os.listdir():
        if os.path.isfile(filename):
                with open(filename, "r+", encoding="utf-8") as f:
                    content = f.read()
                    f.seek(0)  # Move file pointer to the beginning
                    f.write(content.lower())
                    f.truncate()  # Remove any remaining original content



    # Ensure NLTK data is downloaded
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    # Word Tokenization
    for filename in os.listdir(): # يمر على الفايلز
        if os.path.isfile(filename):# يعالج الفايلز فقط

                with open(filename, "r", encoding="utf-8") as f:#يفتحهم ةيقراهم
                    content = f.read()

                # Tokenize words
                tokens = word_tokenize(content) # Fixed: Corrected indentation to align with the 'try' block.


    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words("english"))

    # يمر على الفايلز
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(".txt"):
            with open(file_path, "r+", encoding="utf-8") as file:

                file_content = file.read()

                filtered_content = " ".join(word for word in file_content.split() if word.lower() not in stop_words) # Remove stop words


                file.seek(0)
                file.write(filtered_content)
                file.truncate()  # Remove any remaining original content




    nltk.download("punkt")
    stemmer = SnowballStemmer("english")

    def apply_stemming(text):
        words = nltk.word_tokenize(text)
        stemmed_words = [stemmer.stem(word) for word in words]
        return " ".join(stemmed_words)

    folder_path = "/content/drive/MyDrive/dm_project_cleaned"
    stemmed_folder_path = "/content/drive/MyDrive/dm_project_stemmed"
    os.makedirs(stemmed_folder_path, exist_ok=True)

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)

            with open(file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()

            stemmed_lines = [apply_stemming(line.strip()) for line in lines]

            stemmed_file_path = os.path.join(stemmed_folder_path, file_name)
            with open(stemmed_file_path, "w", encoding="utf-8") as stemmed_file:
                stemmed_file.write("\n".join(stemmed_lines))

    print("All files have been stemmed and saved in:", stemmed_folder_path)




    folder_path ="/content/drive/MyDrive/dm_project_stemmed"
    directory = folder_path

    documents = []

    for filename in os.listdir(directory):

        if filename.endswith('.txt'):  #يتاكد من نوع الملف الي نبيه
            file_path = os.path.join(directory, filename)  # Full path to the file

            # Check if it's a valid file
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    # Read the content of the file and append it to the documents list
                    documents.append(file.read())  # Read the entire file content
                    print(f"Successfully read {filename}")# will print the files that have been read!!


    vectorizer = TfidfVectorizer()  #الي بيسوي لنا المايتركس  tf-idf
    tfidf_matrix = vectorizer.fit_transform(documents)  # ندخله على الدوكمنتس

    # يجيب التيرمز
    feature_names = vectorizer.get_feature_names_out()

    # Fit and transform the text data into a TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Convert TF-IDF matrix to a dense array and print it
    tfidf_array = tfidf_matrix.toarray()
    feature_names = vectorizer.get_feature_names_out()

    # Convert TF-IDF matrix to a Pandas DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # Print the DataFrame to see words instead of numbers

    print(tfidf_df)
    print("\nTF-IDF Matrix:")
    print(tfidf_matrix.toarray())


    # المسار إلى الملفات
    folder_path ="/content/drive/MyDrive/dm_project_stemmed"# أو أي مسار حفظت فيه الملفات
    file_list = sorted(os.listdir(folder_path))
    documents = []

    for file_name in file_list:
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
                documents.append(file.read())

    # الآن، هذه قائمة أسماء الملفات (المعاني) بالترتيب
    document_names = [file_name for file_name in file_list if file_name.endswith(".txt")]



    # Step 1: Define your query (choose something meaningful from your topic)
    query = "delicious food and good service "  # or try a book-related one

    # Step 2: Transform the query using the SAME vectorizer
    query_vector = vectorizer.transform([query])

    # Step 3: Compute cosine similarity between the query and all documents
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Step 4: Get top matches (optional: sort and print top 5 similar docs)
    top_indices = similarities.argsort()[::-1]  # Sort by descending similarity

    print("\nTop matching documents for the query:")
    for idx in top_indices[:30]:
        score = similarities[idx]
        if score > 0:
            print(f"{document_names[idx]} (Score: {score:.4f})")

    # Install Gradio


    # Load image
    logo = Image.open("/content/IMG_A57A1B33449D-1.jpeg")

    # Install Gradio


    # Load image
    logo = Image.open("/content/IMG_A57A1B33449D-1.jpeg")


    # Define the search function using TF-IDF and cosine similarity
    def search_documents(query):
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[::-1]
        results = []
        for idx in top_indices[:5]:  # Show top 5 results only
            score = similarities[idx]
            if score > 0:
                results.append(f"{document_names[idx]} (Score: {score:.4f})")
        return "\n".join(results)

    # Build the Gradio interface
    with gr.Blocks() as demo:
        # Display logo at the top with medium size
        gr.Image(value=logo, label="", show_label=False, show_download_button=False, height=150, width=150)

        # Heading and input box
        gr.Markdown("## Enter your query to retrieve the most similar documents")

        query_input = gr.Textbox(lines=2, placeholder="Type your query here")
        output_text = gr.Textbox(label="Top related Documents")

        # Search button
        search_button = gr.Button("Search")
        search_button.click(fn=search_documents, inputs=query_input, outputs=output_text)

    # Launch the interface
    demo.launch(share=True, debug=True)

if __name__ == '__main__':
    main()
