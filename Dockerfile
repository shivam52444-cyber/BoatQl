FROM python:3.11-slim

WORKDIR /app

# requirements copied and installed BEFORE the rest of the code, so this
# layer stays cached across rebuilds unless requirements.txt itself changes
# -- code-only changes won't trigger a full dependency reinstall.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]