# 🎭 Tiyatro Oyunu Chatbot - RAG Uygulaması

Tiyatro oyunu hakkında bilgi veren ve kullanıcı sorularını yanıtlayan akıllı chatbot uygulaması. LangChain RAG (Retrieval-Augmented Generation) mimarisi kullanılarak geliştirilmiştir.
## Çalışır Haldeki Video

https://github.com/user-attachments/assets/a254545b-1541-4a8c-b908-b72de53d988b


## Proje Hakkında

Bu proje, bir tiyatro oyununun PDF dosyasından bilgi alarak kullanıcı sorularına cevap veren bir chatbot uygulamasıdır. İki farklı LLM modeli (Gemini ve HuggingFace) desteklemektedir ve kullanıcılar arayüzden model seçimi yapabilir.

## Özellikler

- **Çoklu Model Desteği**: Google Gemini 2.5 Flash Lite ve HuggingFace modelleri
- **PDF Tabanlı Bilgi**: Tiyatro oyunu PDF'inden otomatik bilgi çıkarımı
- **RAG Mimarisi**: LangChain ile gelişmiş retrieval-augmented generation
- **Verimli Embedding**: Tek seferlik embedding oluşturma, tekrar kullanma
- **Modern Arayüz**: Streamlit ile kullanıcı dostu chat arayüzü
- **Ortak Veritabanı**: Her iki model için paylaşımlı ChromaDB vektör veritabanı

## Mimari

```
┌─────────────────┐
│  Streamlit UI   │
└────────┬────────┘
         │
    ┌────┴────┐
    │ Models  │
    ├─────────┤
    │ Gemini  │
    │ HFace   │
    └────┬────┘
         │
┌────────┴────────┐
│  ChromaDB       │
│  HuggingFace    │
│  Embeddings     │
└─────────────────┘
         │
┌────────┴────────┐
│   PDF Kaynak    │
│ (Tiyatro Oyunu) │
└─────────────────┘
```

## Proje Yapısı

```
mth-term-project/
├── app/
│   └── streamlit_app.py      # Ana Streamlit uygulaması
├── models/
│   ├── gemini_model.py        # Gemini model implementasyonu
│   └── hface_model.py         # HuggingFace model (şu anda çalışmıyor)
├── data/
│   └── sonkartus.pdf          # Tiyatro oyunu PDF dosyası
├── embed_data.py              # Embedding oluşturma scripti
├── chroma_db/                 # Vektör veritabanı (otomatik oluşturulur)
├── requirements.txt           # Python bağımlılıkları
└── README.md
```

## Kurulum

### 1. Projeyi Klonlayın
```bash
git clone <repo-url>
cd mth-term-project
```

### 2. Sanal Ortam Oluşturun
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# veya
venv\Scripts\activate     # Windows
```

### 3. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 4. API Anahtarlarını Ayarlayın
`.env` dosyası oluşturun:
```env
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

### 5. Uygulamayı Başlatın
```bash
streamlit run app/streamlit_app.py
```

## Kullanım

1. Tarayıcınızda `http://localhost:8501` adresini açın
2. Sidebar'dan model seçin (Gemini veya HuggingFace)
3. Chat kutusuna tiyatro oyunu hakkında sorular yazın
4. Model yanıtını anında görün

## Teknik Detaylar

### Kullanılan Teknolojiler
- **LangChain**: RAG pipeline orchestration
- **ChromaDB**: Vektör veritabanı
- **Streamlit**: Web arayüzü
- **Google Gemini 2.5 Flash Lite**: Ana LLM modeli
- **HuggingFace Embeddings**: Metin vektörizasyonu
- **PyPDF**: PDF okuma

### Model Seçimi

**Gemini 2.5 Flash Lite** kullanılmasının nedenleri:
- ✅ Yüksek ücretsiz API kotası
- ✅ Hızlı yanıt süresi
- ✅ Türkçe dil desteği
- ✅ RAG için uygun performans

### Embedding Stratejisi

**HuggingFace Embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) tercih edildi:
- ✅ Tamamen ücretsiz
- ✅ Kota sınırı yok
- ✅ Lokal çalışma seçeneği
- ✅ Her iki model için uyumlu

Google Gemini Embeddings'de kota sorunları yaşandığından HuggingFace alternatifi kullanıldı.

### Optimizasyonlar

1. **Ayrı Embedding Scripti**: `embed_data.py` ile embedding işlemi bir kere yapılır, model kodlarından ayrıdır
2. **Ortak Veritabanı**: Aynı embedding kullanıldığı için tek ChromaDB her iki model tarafından paylaşılır
4. **Chunk Optimizasyonu**: 1000 karakter chunk size ile optimal retrieval

## Bilinen Sorunlar

- **HuggingFace Model**: HuggingFace Inference API'de provider uyumluluk sorunları nedeniyle şu anda kullanılamıyor
- Alternatif olarak sadece Gemini modeli aktif kullanılabilir

## Gelecek Geliştirmeler

- [ ] HuggingFace API sorunlarının çözümü
- [ ] Chat geçmişi özelliği
- [ ] Çoklu PDF desteği
- [ ] Performans metrikleri (RAGAS entegrasyonu)
- [ ] Daha fazla LLM model seçeneği
