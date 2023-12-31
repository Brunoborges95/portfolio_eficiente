# Use uma imagem base leve com Python
FROM python:3.11-slim

# Defina o diretório de trabalho
WORKDIR /app

# Copie os arquivos necessários para o contêiner
COPY . /app

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Defina as variáveis de ambiente
ENV AWS_ACCESS_KEY_ID = ${{ secrets.AWS_ACCESS_KEY_ID }}
ENV AWS_SECRET_ACCESS_KEY = ${{ secrets.AWS_SECRET_ACCESS_KEY }}

# Exponha a porta utilizada pelo Streamlit
EXPOSE 8501

# Comando para executar o aplicativo Streamlit
CMD streamlit run app.py --server.port $PORT
