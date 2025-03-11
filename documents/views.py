from django.shortcuts import render, redirect, get_object_or_404
from .models import Document
from .utils import (
    extract_title, extract_details,
    calculate_similarity, perform_clustering, ekstrak_dokumen, ekstrak_komponen
)
import networkx as nx
from django.http import JsonResponse
import networkx as nx
from .models import Komponen, Document , Dokumen, DokumenKomponen, KeterkaitanDokumen, KeterkaitanKomponen
from .services import get_documents, delete_document  
import pandas as pd
import numpy as np
from django.http import JsonResponse
import fitz
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.contrib import messages
from django.db.models import Q

def delete_document_view(request, doc_id):
    if request.method == 'POST':
        delete_document(doc_id)
        if 'similarity_results' in request.session:
            del request.session['similarity_results']
        if 'similarity_matrix' in request.session:
            del request.session['similarity_matrix']
        if 'graph_nodes' in request.session:
            del request.session['graph_nodes']
        if 'graph_edges' in request.session:
            del request.session['graph_edges']
        update_similarity_session(request)
        return redirect('home')
    return redirect('home')

def home(request):
    documents = get_documents()
    if request.method == 'POST' and 'delete' in request.POST:
        doc_id = request.POST['delete']
        delete_document(doc_id)
        return redirect('home')
    return render(request, 'home.html', {'documents': documents})

def view_document(request, doc_id):
    doc = Document.objects.get(id=doc_id)
    return render(request, "view_document.html", {'doc': doc})

def pdf_to_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    word_tokens = word_tokenize(text)
    filtered_tokens = [word for word in word_tokens if word.isalnum() or '-' in word]
    cleaned_text = " ".join(filtered_tokens)
    return cleaned_text

def add_document(request):
    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        content = pdf_to_text(file)
        title = extract_title(content)
        details = extract_details(content)
        Document.objects.create(
            title=title,
            pemrakarsa=details.get('Pemrakarsa', ''),
            level_peraturan=details.get('Level Peraturan', ''),
            konten_penimbang=details.get('Konten Penimbang', ''),
            peraturan_terkait=details.get('Peraturan Terkait', ''),
            konten_peraturan=details.get('Konten Peraturan', ''),
            kategori_peraturan=details.get('Kategori Peraturan', ''),
            topik_peraturan=details.get('Topik Peraturan', ''),
            struktur_peraturan=details.get('Struktur Peraturan', '')
        )
        update_similarity_session(request)
        return redirect('home')
    return render(request, "add_document.html")

def similarity_graph(request):
    if 'graph_nodes' in request.session and 'graph_edges' in request.session:
        nodes = request.session['graph_nodes']
        edges = request.session['graph_edges']
    else:
        nodes, edges = update_similarity_session(request)[2], []
    # print("nodes and edges:", nodes, edges)
    return JsonResponse({"nodes": nodes, "edges": edges})

def update_similarity_session(request):
    documents = Document.objects.all()
    if len(documents) > 1:
        similarity_results, similarity_matrix = calculate_similarity(documents)
        similarity_matrix = similarity_matrix.astype(float)
        request.session['similarity_results'] = similarity_results
        request.session['similarity_matrix'] = similarity_matrix.tolist()
        G = nx.Graph()
        titles = [doc.title for doc in documents]
        for i, title in enumerate(titles):
            G.add_node(i, label=title)
        threshold = 0.1
        edges = []
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity_score = similarity_matrix[i, j]
                if similarity_score > threshold:
                    similarity_score = min(similarity_score, 100)
                    normalized_width = similarity_score / 10
                    G.add_edge(i, j, weight=similarity_score)
                    edges.append({
                        "from": i,
                        "to": j,
                        "width": normalized_width,
                        "label": f"{similarity_score:.2f}%",
                    })
        request.session['graph_nodes'] = [{"id": i, "label": title} for i, title in enumerate(titles)]
        request.session['graph_edges'] = edges
        return similarity_results, similarity_matrix, edges
    return [], [], []

def similarity(request):
    similarity_results = request.session.get('similarity_results', None)
    # print("Similarity Results:", similarity_results)
    clustering_ready = bool(similarity_results)
    return render(request, "similarity.html", {'clustering_ready': clustering_ready})

def clustering(request):
    documents = Document.objects.all()
    similarity_matrix = request.session.get('similarity_matrix')
    if not similarity_matrix:
        return redirect('similarity')
    similarity_matrix = np.array(similarity_matrix)
    silhouette_avg, labels = perform_clustering(similarity_matrix, num_clusters=2)
    cluster_data = {f"Cluster {i + 1}": [] for i in range(2)}
    for i, label in enumerate(labels):
        cluster_data[f"Cluster {label + 1}"].append(documents[i].title)
    max_cluster_length = max(len(cluster) for cluster in cluster_data.values())
    for key in cluster_data:
        cluster_data[key].extend([''] * (max_cluster_length - len(cluster_data[key])))
    cluster_df = pd.DataFrame(cluster_data)
    context = {
        'clusters': cluster_df.to_html(classes="table table-striped", index=False),
        'silhouette_avg': silhouette_avg
    }
    return render(request, "clustering.html", context)

def similarity_detail(request, i, j):
    similarity_results = request.session.get('similarity_results', [])
    if not similarity_results:
        return render(request, "error.html", {"message": "Similarity results not found in session"})
    detail = next(
        (result["detail_similarity"] for result in similarity_results if result["detail_url"] == f"/similarity_detail/{i}/{j}/"),
        None
    )
    if not detail:
        return render(request, "error.html", {"message": "Detail keterkaitan tidak ditemukan"})
    documents = list(Document.objects.all())
    if i >= len(documents) or j >= len(documents):
        return render(request, "error.html", {"message": "Dokumen tidak ditemukan"})
    total_similarity = sum(detail.values()) / len(detail)
    context = {
        "dokumen1": documents[i].title,
        "dokumen2": documents[j].title,
        "detail_similarity": detail,
        "total_similarity": total_similarity,
    }
    return render(request, "similarity_detail.html", context)

def home_new(request):
    query = request.GET.get('q', '').strip()  # Ambil query dari input pencarian
    
    if query:
        documents = Dokumen.objects.filter(title__icontains=query)  # Filter berdasarkan title
        no_results = not documents.exists()  # Cek jika tidak ada hasil
    else:
        documents = Dokumen.objects.all()  # Jika kosong, tampilkan semua dokumen
        no_results = False  # Jangan tampilkan pesan "tidak ditemukan"

    if request.method == 'POST' and 'delete' in request.POST:
        doc_id = request.POST['delete']
        
        try:
            dokumen = Dokumen.objects.get(id=doc_id)

            # Hapus semua komponen terkait di DokumenKomponen
            DokumenKomponen.objects.filter(dokumen=dokumen).delete()
            
            # Hapus semua keterkaitan komponen terkait dengan keterkaitan dokumen
            keterkaitan_dokumen = KeterkaitanDokumen.objects.filter(dok_1=dokumen) | KeterkaitanDokumen.objects.filter(dok_2=dokumen)
            KeterkaitanKomponen.objects.filter(keterkaitan_dokumen__in=keterkaitan_dokumen).delete()
            
            # Hapus semua keterkaitan di KeterkaitanDokumen
            keterkaitan_dokumen.delete()
            
            # Hapus dokumen utama
            dokumen.delete()
            
        except Dokumen.DoesNotExist:
            pass  # Jika dokumen tidak ditemukan, abaikan
            
        return redirect('home_new')  # Kembali ke halaman utama setelah hapus dokumen

    return render(request, 'home_new.html', {'documents': documents, 'query': query, 'no_results': no_results})

def delete_document_view_new(request, doc_id):
    if request.method == 'POST':
        try:
            # Hapus keterkaitan komponen yang terkait dengan dokumen ini
            KeterkaitanKomponen.objects.filter(keterkaitan_dokumen__dok_1_id=doc_id).delete()
            KeterkaitanKomponen.objects.filter(keterkaitan_dokumen__dok_2_id=doc_id).delete()

            # Hapus keterkaitan dokumen yang terkait dengan dokumen ini
            KeterkaitanDokumen.objects.filter(dok_1_id=doc_id).delete()
            KeterkaitanDokumen.objects.filter(dok_2_id=doc_id).delete()

            # Hapus semua komponen yang terkait dengan dokumen ini
            DokumenKomponen.objects.filter(dokumen_id=doc_id).delete()

            # Hapus dokumen itu sendiri
            Dokumen.objects.filter(id=doc_id).delete()

            return redirect('home')
        except Exception as e:
            print(f"Error saat menghapus dokumen: {e}")

    return redirect('home')

def add_document_new(request):
    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']

        content = pdf_to_text(file)

        dokumen = ekstrak_dokumen(content)

        ekstrak_komponen(content, dokumen)

        # **Hitung Similarity**
        calculate_component_similarity()

        # **Cek apakah ada dokumen dengan similarity ≥ 1.0**
        similarity_exists = KeterkaitanDokumen.objects.filter(nilai_total_keterkaitan__gte=1.0).exists()

        if similarity_exists:
            dokumen.delete()  # Hapus dokumen baru karena sudah ada yang sama
            messages.error(request, "Dokumen sudah ada dalam database.")
            return render(request, "add_document.html")

        # **Jika tidak ada yang sama, simpan dokumen**
        messages.success(request, "Dokumen berhasil ditambahkan.")
        return redirect('home')

    return render(request, "add_document.html")

def view_document_new(request, doc_id):
    # Ambil dokumen berdasarkan ID
    doc = get_object_or_404(Dokumen, id=doc_id)

    # Ambil semua komponen yang terkait dengan dokumen ini
    komponen_list = DokumenKomponen.objects.filter(dokumen=doc)

    return render(
        request, 
        "view_document_new.html", 
        {'doc': doc, 'komponen_list': komponen_list}
    )

def calculate_component_similarity():
    dokumen_komponen = DokumenKomponen.objects.all()
    dokumen_dict = {}

    # Mengelompokkan komponen berdasarkan dokumen
    for dk in dokumen_komponen:
        if dk.dokumen.id not in dokumen_dict:
            dokumen_dict[dk.dokumen.id] = {}
        dokumen_dict[dk.dokumen.id][dk.komponen.id] = dk.isi_komponen

    dokumen_ids = list(dokumen_dict.keys())
    komponen_ids = list(Komponen.objects.values_list('id', flat=True))
    
    # Menghitung similarity per komponen
    for i in range(len(dokumen_ids)):
        for j in range(i + 1, len(dokumen_ids)):
            dok_1_id, dok_2_id = dokumen_ids[i], dokumen_ids[j]
            keterkaitan_dokumen, _ = KeterkaitanDokumen.objects.get_or_create(
                dok_1_id=dok_1_id, dok_2_id=dok_2_id, defaults={'nilai_total_keterkaitan': 0}
            )

            total_similarity = []
            for komponen_id in komponen_ids:
                isi_1 = dokumen_dict.get(dok_1_id, {}).get(komponen_id, "")
                isi_2 = dokumen_dict.get(dok_2_id, {}).get(komponen_id, "")

                if isi_1 and isi_2:
                    tfidf = TfidfVectorizer().fit_transform([isi_1, isi_2])
                    similarity_score = cosine_similarity(tfidf)[0, 1]
                    total_similarity.append(similarity_score)
                    
                    # Simpan hasil keterkaitan komponen
                    KeterkaitanKomponen.objects.update_or_create(
                        keterkaitan_dokumen=keterkaitan_dokumen,
                        komponen_id=komponen_id,
                        defaults={'nilai_keterkaitan_komponen': similarity_score}
                    )

            # Simpan total keterkaitan dokumen
            if total_similarity:
                keterkaitan_dokumen.nilai_total_keterkaitan = np.mean(total_similarity)
                keterkaitan_dokumen.save()

def similarity_graph_new(request):
    keterkaitan_list = KeterkaitanDokumen.objects.select_related('dok_1', 'dok_2').all()

    # Jika tidak ada data, kembalikan response kosong
    if not keterkaitan_list.exists():
        return JsonResponse({"nodes": [], "edges": []})

    nodes = {}
    edges = []

    # Ambil semua dokumen dan simpan dalam dictionary
    dokumen_dict = {dok.id: dok.title for dok in Dokumen.objects.all()}

    # Tetapkan threshold minimum untuk menampilkan koneksi (50% ke atas)
    threshold = 50  

    for keterkaitan in keterkaitan_list:
        dok_1_id, dok_2_id = keterkaitan.dok_1.id, keterkaitan.dok_2.id
        nilai_keterkaitan = keterkaitan.nilai_total_keterkaitan * 100  # Ubah ke persen

        # Hanya proses jika keterkaitan >= 50%
        if nilai_keterkaitan >= threshold:
            # Tambahkan node jika belum ada
            if dok_1_id not in nodes:
                nodes[dok_1_id] = {"id": dok_1_id, "label": dokumen_dict.get(dok_1_id, "Unknown")}
            if dok_2_id not in nodes:
                nodes[dok_2_id] = {"id": dok_2_id, "label": dokumen_dict.get(dok_2_id, "Unknown")}

            # Tambahkan edge dengan lebar minimal 1
            edges.append({
                "from": dok_1_id,
                "to": dok_2_id,
                "width": max(1, nilai_keterkaitan * 0.15),  # Pastikan width minimal 1
                "label": f"{nilai_keterkaitan:.0f}%",  # Format persen tanpa desimal
            })

    return JsonResponse({"nodes": list(nodes.values()), "edges": edges})

def similarity_new(request):
    dokumen_list = Dokumen.objects.all()
    return render(request, "similarity_new.html", {"dokumen_list": dokumen_list})

def similarity_detail_new(request, i, j):
    # Ambil dua dokumen berdasarkan ID
    dokumen1 = get_object_or_404(Dokumen, id=i)
    dokumen2 = get_object_or_404(Dokumen, id=j)

    # Cek keterkaitan dokumen di database (tanpa mempermasalahkan urutan)
    keterkaitan = KeterkaitanDokumen.objects.filter(
        (Q(dok_1=dokumen1) & Q(dok_2=dokumen2)) | 
        (Q(dok_1=dokumen2) & Q(dok_2=dokumen1))
    ).first()
    
    if not keterkaitan:
        return render(request, "error.html", {"message": "Detail keterkaitan tidak ditemukan dalam database"})

    # Ambil detail similarity per komponen
    komponen_similarity = KeterkaitanKomponen.objects.filter(keterkaitan_dokumen=keterkaitan)

    # Buat dictionary untuk detail similarity dalam persen
    detail_similarity = {
        k.komponen.nama_komponen: k.nilai_keterkaitan_komponen * 100 for k in komponen_similarity
    }

    context = {
        "dokumen1": dokumen1.title,
        "dokumen2": dokumen2.title,
        "detail_similarity": detail_similarity,
        "total_similarity": keterkaitan.nilai_total_keterkaitan * 100,  # Ubah ke persen
    }
    return render(request, "similarity_detail.html", context)

def similarity_graph_detail(request, doc_id):
    keterkaitan_list = KeterkaitanDokumen.objects.filter(
        Q(dok_1_id=doc_id) | Q(dok_2_id=doc_id)
    ).select_related('dok_1', 'dok_2')

    if not keterkaitan_list.exists():
        return JsonResponse({"nodes": [], "edges": []})

    nodes = {}
    edges = []
    dokumen_dict = {dok.id: dok.title for dok in Dokumen.objects.all()}

    # Warna khusus untuk dokumen utama
    nodes[doc_id] = {"id": doc_id, "label": dokumen_dict.get(doc_id, "Unknown"), "color": "#e74c3c"}

    # Tetapkan threshold minimum (50%)
    threshold = 50  

    for keterkaitan in keterkaitan_list:
        dok_1_id, dok_2_id = keterkaitan.dok_1.id, keterkaitan.dok_2.id
        nilai_keterkaitan = keterkaitan.nilai_total_keterkaitan * 100  # Ubah ke persen

        # Pastikan kita mendapatkan dokumen yang bukan dirinya sendiri
        related_doc_id = dok_2_id if dok_1_id == doc_id else dok_1_id

        # Hanya tampilkan jika keterkaitan >= 50%
        if nilai_keterkaitan >= threshold:
            # Tambahkan node jika belum ada
            if related_doc_id not in nodes:
                nodes[related_doc_id] = {
                    "id": related_doc_id,
                    "label": dokumen_dict.get(related_doc_id, "Unknown"),
                    "color": "#2ecc71"
                }

            # Tambahkan edge dengan lebar minimal 1
            edges.append({
                "from": doc_id,
                "to": related_doc_id,
                "width": max(1, nilai_keterkaitan * 0.15),  # Pastikan width minimal 1
                "label": f"{nilai_keterkaitan:.0f}%",  # Format persen tanpa desimal
            })

    return JsonResponse({"nodes": list(nodes.values()), "edges": edges})


def clustering_new(request):
    documents = list(Dokumen.objects.all())
    if not documents:
        return render(request, "clustering.html", {"error": "Tidak ada dokumen yang tersedia."})
    
    # Hitung similarity menggunakan fungsi yang sudah ada
    calculate_component_similarity()
    
    # Ambil matriks keterkaitan dari database
    dokumen_ids = [doc.id for doc in documents]
    similarity_matrix = np.zeros((len(dokumen_ids), len(dokumen_ids)))
    
    for i, dok_1 in enumerate(dokumen_ids):
        for j, dok_2 in enumerate(dokumen_ids):
            if dok_1 != dok_2:
                keterkaitan = KeterkaitanDokumen.objects.filter(dok_1_id=dok_1, dok_2_id=dok_2).first()
                if keterkaitan:
                    similarity_matrix[i, j] = keterkaitan.nilai_total_keterkaitan
    
    silhouette_avg, labels = perform_clustering(similarity_matrix, num_clusters=2)
    
    cluster_data = {f"Cluster {i + 1}": [] for i in range(2)}
    for i, label in enumerate(labels):
        cluster_data[f"Cluster {label + 1}"].append(documents[i].title)
    
    max_cluster_length = max(len(cluster) for cluster in cluster_data.values())
    for key in cluster_data:
        cluster_data[key].extend([''] * (max_cluster_length - len(cluster_data[key])))
    
    cluster_df = pd.DataFrame(cluster_data)
    context = {
        'clusters': cluster_df.to_html(classes="table table-striped", index=False),
        'silhouette_avg': silhouette_avg
    }
    return render(request, "clustering.html", context)