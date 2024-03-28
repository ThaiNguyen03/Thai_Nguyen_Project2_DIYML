
FROM python:3.12


WORKDIR /app


ADD . /app


RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "Reports.py"]
