import psycopg2
def insert_pdf(filename, pdf_path):
    try:
        connection = psycopg2.connect(
            dbname="ragpdf",
            user="postgres",
            password="123",
            host="localhost",
            port="5432"
        )

        cursor = connection.cursor()
        with open(pdf_path, 'rb') as file:
            file_data = file.read()

        cursor.execute(
            "INSERT INTO pdf_files (filename, file_data) VALUES (%s, %s)",
            (filename, psycopg2.Binary(file_data))
        )

        connection.commit()
        print("PDF inserted successfully")

    except Exception as e:
        print(f"Error inserting PDF: {e}")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

insert_pdf('test.pdf', 'test.pdf')
