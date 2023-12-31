# Use uma imagem base leve com Python
FROM python:3.11-slim

# Defina o diretório de trabalho
WORKDIR /app

# Copie os arquivos necessários para o contêiner
COPY . /app

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Exponha a porta utilizada pelo Streamlit
EXPOSE 8501

# Comando para executar o aplicativo Streamlit
CMD streamlit run app.py --server.port $PORT
