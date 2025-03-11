# services.py atau views.py
from .models import Document

def save_document(title, details):
    document = Document(
        title=title,
        pemrakarsa=details['Pemrakarsa'],
        level_peraturan=details['Level Peraturan'],
        konten_penimbang=details['Konten Penimbang'],
        peraturan_terkait=details['Peraturan Terkait'],
        konten_peraturan=details['Konten Peraturan'],
        kategori_peraturan=details['Kategori Peraturan'],
        topik_peraturan=details['Topik Peraturan'],
        struktur_peraturan=details['Struktur Peraturan']
    )
    document.save()


# Fungsi untuk mengambil semua dokumen dari database
def get_documents():
    return Document.objects.all()

# Fungsi untuk menghapus dokumen berdasarkan ID
def delete_document(doc_id):
    try:
        doc = Document.objects.get(id=doc_id)
        doc.delete()
    except Document.DoesNotExist:
        pass  # Jika dokumen tidak ditemukan, biarkan saja
