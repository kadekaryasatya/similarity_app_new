import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
from .models import Dokumen , Komponen, DokumenKomponen

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

# case folding, tokenizing
def preprocess_text(text):
    # Tokenisasi
    word_tokens = word_tokenize(text)
    
    # Menghapus token non-alfanumerik
    clean_tokens = [word for word in word_tokens if word.isalnum()]

    # Lowercase dan Menghapus stop words 
    filtered_text = [word.lower() for word in clean_tokens if word.lower()]

    cleaned_text = " ".join(filtered_text)

    return cleaned_text

def ekstrak_dokumen(text):
    # Ekstrak judul dokumen
    title_pattern = r'\b(Peraturan|Undang-Undang)\b.*?(?=\bdengan rahmat\b)'
    title_match = re.search(title_pattern, text, re.IGNORECASE | re.DOTALL)
    title = title_match.group().strip() if title_match else "Judul Tidak Ditemukan"

    penandatangan_pattern = r'ttd[\s\n]+(\S+\s+\S+)'
    penandatangan_match = re.search(penandatangan_pattern, text, re.IGNORECASE)
    penandatangan = penandatangan_match.group(1).strip() if penandatangan_match else "Penandatangan Tidak Ditemukan"

    # Ekstrak tanggal ditandatangani (tiga kata setelah "pada tanggal")
    tanggal_pattern = r'pada tanggal[\s\n]+(\S+\s+\S+\s+\S+)'
    tanggal_matches = re.findall(tanggal_pattern, text, re.IGNORECASE)
    tanggal_ditandatangani = tanggal_matches[-1].strip() if tanggal_matches else "Tanggal Tidak Ditemukan"
    
    lembaga_pattern = r'\b(?:Presiden|Wakil Presiden|Lembaga Pemerintah Non-Kementerian|Dewan Perwakilan Rakyat(?: \(DPR\))?|Majelis Permusyawaratan Rakyat(?: \(MPR\))?|Mahkamah Agung(?: \(MA\))?|Mahkamah Konstitusi(?: \(MK\))?|Badan Pemeriksa Keuangan(?: \(BPK\))?|Bank Indonesia(?: \(BI\))?|Otoritas Jasa Keuangan(?: \(OJK\))?|Badan Pengawas Pemilihan Umum(?: \(Bawaslu\))?|Komisi Pemilihan Umum(?: \(KPU\))?|Komisi Pemberantasan Korupsi(?: \(KPK\))?|Gubernur|Dewan Perwakilan Rakyat Daerah Provinsi(?: \(DPRD Provinsi\))?|Bupati|Walikota|Dewan Perwakilan Rakyat Daerah Kabupaten/Kota(?: \(DPRD Kabupaten/Kota\))?|Kepala Desa|Lurah|Badan Permusyawaratan Desa(?: \(BPD\))?|Komisi Yudisial(?: \(KY\))?|Lembaga Negara Independen|Komnas HAM|Komisi Informasi|Menteri(?: [A-Za-z]+)*)\b'
    lembaga_match = re.search(lembaga_pattern, text, re.IGNORECASE)
    lembaga = lembaga_match.group(0).strip() if lembaga_match else "Lembaga tidak ditemukan"

    # Ekstrak nomor dokumen
    no_dokumen_pattern = r'No\.\s*(\d+/\d+/\d+)'
    no_dokumen_match = re.search(no_dokumen_pattern, text)
    no_dokumen = no_dokumen_match.group(1) if no_dokumen_match else "Nomor Tidak Ditemukan"

    # Ekstrak status dokumen
    status_pattern = r'(berlaku|dicabut|diubah|revisi)'
    status_match = re.search(status_pattern, text, re.IGNORECASE)
    status_dokumen = status_match.group(0).capitalize() if status_match else "Status Tidak Ditemukan"

    # Ekstrak lokasi penerbit
    lokasi_pattern = r'ditetapkan di\s+(\S+)'
    lokasi_match = re.search(lokasi_pattern, text, re.IGNORECASE)
    lokasi_penerbit = lokasi_match.group(1) if lokasi_match else "Lokasi Tidak Ditemukan"

    # Ekstrak sumber
    sumber_pattern = r'(lembaran negara|berita negara|jurnal resmi)'  
    sumber_match = re.search(sumber_pattern, text, re.IGNORECASE)
    sumber = sumber_match.group(0).capitalize() if sumber_match else "Sumber Tidak Ditemukan"

    # Ekstrak ringkasan (ambil paragraf pertama)
    ringkasan_pattern = r'\n(.+?)(?=\n\n)'
    ringkasan_match = re.search(ringkasan_pattern, text, re.DOTALL)
    ringkasan = ringkasan_match.group(1).strip() if ringkasan_match else "Ringkasan Tidak Ditemukan"

    # Simpan ke database
    dokumen = Dokumen.objects.create(
        title=title,
        penandatangan=penandatangan,
        tanggal_ditandatangani=tanggal_ditandatangani,
        lembaga_mengeluarkan=lembaga,
        no_dokumen=no_dokumen,
        status_dokumen=status_dokumen,
        lokasi_penerbit=lokasi_penerbit,
        sumber=sumber,
        ringkasan=ringkasan
    )
    
    return dokumen

def ekstrak_komponen(text, dokumen):
    # Ekstrak judul dokumen
    title_pattern = r'\b(Peraturan|Undang-Undang)\b.*?(?=\bdengan rahmat\b)'
    title_match = re.search(title_pattern, text, re.IGNORECASE | re.DOTALL)
    title = title_match.group().strip() if title_match else "Judul Tidak Ditemukan"

    pemrakarsa_pattern = r'\b(?:Presiden|Wakil Presiden|Lembaga Pemerintah Non-Kementerian|Dewan Perwakilan Rakyat(?: \(DPR\))?|Majelis Permusyawaratan Rakyat(?: \(MPR\))?|Mahkamah Agung(?: \(MA\))?|Mahkamah Konstitusi(?: \(MK\))?|Badan Pemeriksa Keuangan(?: \(BPK\))?|Bank Indonesia(?: \(BI\))?|Otoritas Jasa Keuangan(?: \(OJK\))?|Badan Pengawas Pemilihan Umum(?: \(Bawaslu\))?|Komisi Pemilihan Umum(?: \(KPU\))?|Komisi Pemberantasan Korupsi(?: \(KPK\))?|Gubernur|Dewan Perwakilan Rakyat Daerah Provinsi(?: \(DPRD Provinsi\))?|Bupati|Walikota|Dewan Perwakilan Rakyat Daerah Kabupaten/Kota(?: \(DPRD Kabupaten/Kota\))?|Kepala Desa|Lurah|Badan Permusyawaratan Desa(?: \(BPD\))?|Komisi Yudisial(?: \(KY\))?|Lembaga Negara Independen|Komnas HAM|Komisi Informasi|Menteri Dalam Negeri|Menteri Luar Negeri|Menteri Pertahanan|Menteri Hukum dan Hak Asasi Manusia|Menteri Keuangan|Menteri Pendidikan dan Kebudayaan|Menteri Riset dan Teknologi|Menteri Agama|Menteri Ketenagakerjaan|Menteri Energi dan Sumber Daya Mineral|Menteri Perindustrian|Menteri Perdagangan|Menteri Pertanian|Menteri Lingkungan Hidup dan Kehutanan|Menteri Kelautan dan Perikanan|Menteri Desa, Pembangunan Daerah Tertinggal, dan Transmigrasi|Menteri Perencanaan Pembangunan Nasional|Menteri Pendayagunaan Aparatur Negara dan Reformasi Birokrasi|Menteri Pekerjaan Umum dan Perumahan Rakyat|Menteri Kesehatan|Menteri Sosial|Menteri Pariwisata|Menteri Komunikasi dan Informatika|Menteri Koordinator Bidang Politik, Hukum, dan Keamanan|Menteri Koordinator Bidang Perekonomian|Menteri Koordinator Bidang Pembangunan Manusia dan Kebudayaan|Menteri Koordinator Bidang Kemaritiman dan Investasi|Menteri Badan Usaha Milik Negara|Menteri Koperasi dan Usaha Kecil dan Menengah|Menteri Pemuda dan Olahraga|Menteri Perhubungan|Menteri Agraria dan Tata Ruang/Badan Pertanahan Nasional|Menteri Perumahan Rakyat|Menteri Percepatan Pembangunan Daerah Tertinggal|Menteri Perencanaan Pembangunan Nasional/Bappenas|Menteri Sekretaris Negara|Menteri Sekretariat Kabinet|Menteri(?: [A-Za-z]+)*|Kementerian(?: [A-Za-z]+)*)\b'
    pemrakarsa_match = re.search(pemrakarsa_pattern, text, re.IGNORECASE)
    pemrakarsa = pemrakarsa_match.group(0).strip() if pemrakarsa_match else "Pemrakarsa tidak ditemukan"
    
    level_peraturan_pattern = r'\b(?:Undang-Undang Dasar 1945|Ketetapan Majelis Permusyawaratan Rakyat|Undang-Undang|Peraturan Pemerintah Pengganti Undang-Undang|Peraturan Pemerintah|Keputusan Presiden|Peraturan Menteri|Peraturan Gubernur|Peraturan Bupati|Majelis Permusyawaratan Rakyat|Dewan Perwakilan Rakyat|Dewan Perwakilan Daerah|Mahkamah Agung|Mahkamah Konstitusi|Badan Pemeriksa Keuangan|Komisi Yudisial|Bank Indonesia|Menteri)\b'

    level_peraturan_match = re.search(level_peraturan_pattern, text, re.IGNORECASE)
    if level_peraturan_match:
        level_peraturan = level_peraturan_match.group(0).strip()
        if level_peraturan.lower() == "peraturan gubernur":
            level_peraturan = "peraturan daerah provinsi"
        elif level_peraturan.lower() == "peraturan bupati":
            level_peraturan = "peraturan daerah kabupaten/Kota"

        level_list = [
            "Undang-Undang Dasar Negara Republik Indonesia Tahun 1945",
            "Ketetapan Majelis Permusyawaratan Rakyat",
            "Undang-Undang/Peraturan Pemerintah Pengganti Undang-Undang",
            "Peraturan Pemerintah",
            "Peraturan Presiden",
            "Peraturan Daerah Provinsi",
            "Peraturan Daerah Kabupaten/Kota"
        ]

        if level_peraturan in level_list:
            kategori_peraturan = "Peraturan Perundang-undangan"
        else:
            kategori_peraturan = "Peraturan Perundang-undangan lainnya"
    else:
        level_peraturan = "Level Peraturan tidak ditemukan"
        kategori_peraturan = "Kategori Peraturan tidak ditemukan"


    penimbang_pattern = r'Menimbang\s*(.*?)(?=Mengingat|$)'
    penimbang_match = re.search(penimbang_pattern, text, re.DOTALL| re.IGNORECASE)
    konten_penimbang = penimbang_match.group(1).strip() if penimbang_match else "Penimbang tidak ditemukan"

    peraturan_terkait_pattern = r'Mengingat\s*(.*?)(?=Memutuskan|$)'
    peraturan_terkait_match = re.search(peraturan_terkait_pattern, text, re.DOTALL| re.IGNORECASE)
    peraturan_terkait = peraturan_terkait_match.group(1).strip() if peraturan_terkait_match else "Peraturan Terkait tidak ditemukan"
    
    konten_peraturan_pattern = r'Memutuskan\s*(.*?)$'
    konten_peraturan_match = re.search(konten_peraturan_pattern, text, re.IGNORECASE | re.DOTALL)
    konten_peraturan = (konten_peraturan_match.group(1)[:255]).strip() if konten_peraturan_match else "Konten peraturan tidak ditemukan"

    topik_kata_kunci = {
        "pendidikan": ["sekolah", "kurikulum", "pengajaran", "siswa", "guru", "pendidikan tinggi", "universitas", "beasiswa"],
        "kesehatan": ["rumah sakit", "dokter", "obat-obatan", "penyakit menular", "vaksinasi", "pelayanan kesehatan", "asuransi kesehatan"],
        "lingkungan hidup": ["polusi udara", "polusi air", "limbah", "konservasi", "hutan", "energi terbarukan", "pengelolaan sampah"],
        "pertanian": ["tanaman", "peternakan", "lahan pertanian", "irigasi", "pupuk", "pestisida", "perlindungan tanaman"],
        "kelautan dan perikanan": ["perikanan", "kelautan", "kemaritiman"],
        "ketenagakerjaan": ["tenaga kerja", "kepegawaian", "cipta kerja", "hukuman disiplin", "atribut", "upah", "aparatur sipil negara", "penerimaan calon praja", "IPDN", "pakaian dinas", "pegawai negeri sipil", "keamanan kerja", "hak-hak pekerja", "serikat pekerja", "gaji", "perlindungan sosial"],
        "perpajakan": ["pajak penghasilan", "pajak pertambahan nilai", "tarif pajak", "penghindaran pajak", "insentif pajak"],
        "investasi": ["pasar modal", "saham", "obligasi", "regulasi investasi", "perlindungan investor", "modal ventura"],
        "transportasi": ["angkutan","jalan", "transportasi umum", "kendaraan bermotor", "bandara", "pelabuhan", "transportasi massal"],
        "keuangan": ["perbankan", "keuangan", "pembayaran", "pembiayaan", "asuransi", "pasar keuangan", "regulasi keuangan", "inflasi", "suku bunga", "anggaran", "anggaran pendapatan", "belanja daerah", "belanja", "harga", "standar harga", "pengelolaan keuangan daerah", "standar harga satuan", "keuangan daerah", "satuan biaya", "harga pasar", "kebijakan moneter"],
        "hak dan kewajiban": ["hak asasi manusia", "hak", "perlindungan perempuan", "pemberdayaan perempuan", "perlindungan anak", "gender", "kesetaraan gender", "pengarusutamaan gender", "responsif gender", "kesenjangan gender", "analisis gender", "perspektif gender", "korban kekerasan", "diskriminasi", "hak untuk hidup", "kebebasan berbicara", "hak beragama", "kepemilikan properti", "paten"],
        "komunikasi": ["komunikasi", "siaran", "penyiaran", "radio", "iklan", "dokumenter", "film"],
        "budaya dan pariwisata": ["desa wisata", "wisata", "pariwisata", "budaya", "kebudayaan"],
        "riset dan teknologi": ["riset dan teknologi", "teknologi", "inovasi"],
        "bencana alam": ["gempa bumi", "gelombang pasang", "gunung meletus", "tsunami", "erupsi", "banjir", "angin puting beliung", "tornado", "kekeringan", "tanah longsor"],
        "ekonomi": ["ekonomi", "industri", "perdagangan", "perdagangan bebas", "produksi barang"],
        "olahraga": ["pemuda dan olahraga", "kejuaraan olahraga", "olahragawan", "keolahragaan", "atlet", "pelatih olahraga", "olahraga"]
    }

    frekuensi_topik = {topik: 0 for topik in topik_kata_kunci}

    for topik, kata_kunci in topik_kata_kunci.items():
        for kata in kata_kunci:
            frekuensi_topik[topik] += text.count(kata)

    topik_peraturan_text = max(frekuensi_topik, key=frekuensi_topik.get)

    if frekuensi_topik[topik_peraturan_text] == 0:
        topik_peraturan_text = "Topik tidak ditemukan"
    
    judul =  extract_title(text)

    struktur_peraturan_pattern = {
    "Judul": judul,
    "Pembukaan": r'(Dengan Rahmat Tuhan Yang Maha Esa.*?)(?=Menimbang)',
    "Batang Tubuh": konten_penimbang + "\n\n" + peraturan_terkait + "\n\n" + konten_peraturan,
    "Penjelasan": r'(Penjelasan atas.*?)(?=Ditetapkan Di)',
    "Penutup": r'(?:Ditetapkan Di).*?(?=Lampiran|$)',
    "Lampiran": r'(Lampiran.*?$)'
}

    struktur_peraturan = {}

    for bagian, pola in struktur_peraturan_pattern.items():
        if bagian == "Batang Tubuh":
            struktur_peraturan[bagian] = pola
        else:
            matches = re.findall(pola, text, re.DOTALL | re.IGNORECASE)
            if matches:
                if bagian == "Lampiran":
                    struktur_peraturan[bagian] = matches[-1].strip()
                else:
                    struktur_peraturan[bagian] = matches[0].strip()
            else:
                struktur_peraturan[bagian] = "Tidak ditemukan"

    struktur_peraturan_mix = "\n\n".join([f"{bagian}: {konten}" for bagian, konten in struktur_peraturan.items()])

    komponen_data = {
    "title": title,
    "pemrakarsa": pemrakarsa,
    "level_peraturan": level_peraturan,
    "konten_penimbang": konten_penimbang,
    "peraturan_terkait": peraturan_terkait,
    "konten_peraturan": konten_peraturan,
    "kategori_peraturan": kategori_peraturan,
    "topik_peraturan": topik_peraturan_text,
    "struktur_peraturan": struktur_peraturan_mix,
    }

    for nama_komponen, isi_komponen in komponen_data.items():
        isi_komponen = isi_komponen if isi_komponen is not None else ""
        komponen_obj, _ = Komponen.objects.get_or_create(nama_komponen=nama_komponen)
        DokumenKomponen.objects.create(dokumen=dokumen, komponen=komponen_obj, isi_komponen=isi_komponen)

    return dokumen 

def extract_title(text):
    title_pattern = r'\b(Peraturan|Undang-Undang)\b.*?(?=\bdengan rahmat\b)'
    title_match = re.search(title_pattern, text, re.IGNORECASE | re.DOTALL)
    
    if title_match:
        title = title_match.group().strip()
    else:
        title = "Judul Tidak Ditemukan"
    
    return title

def extract_details(text):
    pemrakarsa_pattern = r'\b(?:Presiden|Wakil Presiden|Lembaga Pemerintah Non-Kementerian|Dewan Perwakilan Rakyat(?: \(DPR\))?|Majelis Permusyawaratan Rakyat(?: \(MPR\))?|Mahkamah Agung(?: \(MA\))?|Mahkamah Konstitusi(?: \(MK\))?|Badan Pemeriksa Keuangan(?: \(BPK\))?|Bank Indonesia(?: \(BI\))?|Otoritas Jasa Keuangan(?: \(OJK\))?|Badan Pengawas Pemilihan Umum(?: \(Bawaslu\))?|Komisi Pemilihan Umum(?: \(KPU\))?|Komisi Pemberantasan Korupsi(?: \(KPK\))?|Gubernur|Dewan Perwakilan Rakyat Daerah Provinsi(?: \(DPRD Provinsi\))?|Bupati|Walikota|Dewan Perwakilan Rakyat Daerah Kabupaten/Kota(?: \(DPRD Kabupaten/Kota\))?|Kepala Desa|Lurah|Badan Permusyawaratan Desa(?: \(BPD\))?|Komisi Yudisial(?: \(KY\))?|Lembaga Negara Independen|Komnas HAM|Komisi Informasi|Menteri Dalam Negeri|Menteri Luar Negeri|Menteri Pertahanan|Menteri Hukum dan Hak Asasi Manusia|Menteri Keuangan|Menteri Pendidikan dan Kebudayaan|Menteri Riset dan Teknologi|Menteri Agama|Menteri Ketenagakerjaan|Menteri Energi dan Sumber Daya Mineral|Menteri Perindustrian|Menteri Perdagangan|Menteri Pertanian|Menteri Lingkungan Hidup dan Kehutanan|Menteri Kelautan dan Perikanan|Menteri Desa, Pembangunan Daerah Tertinggal, dan Transmigrasi|Menteri Perencanaan Pembangunan Nasional|Menteri Pendayagunaan Aparatur Negara dan Reformasi Birokrasi|Menteri Pekerjaan Umum dan Perumahan Rakyat|Menteri Kesehatan|Menteri Sosial|Menteri Pariwisata|Menteri Komunikasi dan Informatika|Menteri Koordinator Bidang Politik, Hukum, dan Keamanan|Menteri Koordinator Bidang Perekonomian|Menteri Koordinator Bidang Pembangunan Manusia dan Kebudayaan|Menteri Koordinator Bidang Kemaritiman dan Investasi|Menteri Badan Usaha Milik Negara|Menteri Koperasi dan Usaha Kecil dan Menengah|Menteri Pemuda dan Olahraga|Menteri Perhubungan|Menteri Agraria dan Tata Ruang/Badan Pertanahan Nasional|Menteri Perumahan Rakyat|Menteri Percepatan Pembangunan Daerah Tertinggal|Menteri Perencanaan Pembangunan Nasional/Bappenas|Menteri Sekretaris Negara|Menteri Sekretariat Kabinet|Menteri(?: [A-Za-z]+)*|Kementerian(?: [A-Za-z]+)*)\b'
    pemrakarsa_match = re.search(pemrakarsa_pattern, text, re.IGNORECASE)
    pemrakarsa = pemrakarsa_match.group(0).strip() if pemrakarsa_match else "Pemrakarsa tidak ditemukan"
    
    level_peraturan_pattern = r'\b(?:Undang-Undang Dasar 1945|Ketetapan Majelis Permusyawaratan Rakyat|Undang-Undang|Peraturan Pemerintah Pengganti Undang-Undang|Peraturan Pemerintah|Keputusan Presiden|Peraturan Menteri|Peraturan Gubernur|Peraturan Bupati|Majelis Permusyawaratan Rakyat|Dewan Perwakilan Rakyat|Dewan Perwakilan Daerah|Mahkamah Agung|Mahkamah Konstitusi|Badan Pemeriksa Keuangan|Komisi Yudisial|Bank Indonesia|Menteri)\b'
    level_peraturan_match = re.search(level_peraturan_pattern, text, re.IGNORECASE)
    if level_peraturan_match:
        level_peraturan = level_peraturan_match.group(0).strip()
        if level_peraturan.lower() == "peraturan gubernur":
            level_peraturan = "peraturan daerah provinsi"
        elif level_peraturan.lower() == "peraturan bupati":
            level_peraturan = "peraturan daerah kabupaten/Kota"
    else:
        level_peraturan = "Level Peraturan tidak ditemukan"

    penimbang_pattern = r'Menimbang\s*(.*?)(?=Mengingat|$)'
    penimbang_match = re.search(penimbang_pattern, text, re.DOTALL| re.IGNORECASE)
    konten_penimbang = penimbang_match.group(1).strip() if penimbang_match else "Penimbang tidak ditemukan"

    peraturan_terkait_pattern = r'Mengingat\s*(.*?)(?=Memutuskan|$)'
    peraturan_terkait_match = re.search(peraturan_terkait_pattern, text, re.DOTALL| re.IGNORECASE)
    peraturan_terkait = peraturan_terkait_match.group(1).strip() if peraturan_terkait_match else "Peraturan Terkait tidak ditemukan"
    
    konten_peraturan_pattern = r'Memutuskan\s*(.*?)$'
    konten_peraturan_match = re.search(konten_peraturan_pattern, text, re.IGNORECASE | re.DOTALL)
    konten_peraturan = (konten_peraturan_match.group(1)[:255]).strip() if konten_peraturan_match else "Konten peraturan tidak ditemukan"

    if level_peraturan == "Level Peraturan tidak ditemukan":
       kategori_peraturan = "peraturan biasa"
    else:
       kategori_peraturan = "peraturan perundang-undangan"
    
    topik_kata_kunci = {
        "pendidikan": ["sekolah", "kurikulum", "pengajaran", "siswa", "guru", "pendidikan tinggi", "universitas", "beasiswa"],
        "kesehatan": ["rumah sakit", "dokter", "obat-obatan", "penyakit menular", "vaksinasi", "pelayanan kesehatan", "asuransi kesehatan"],
        "lingkungan hidup": ["polusi udara", "polusi air", "limbah", "konservasi", "hutan", "energi terbarukan", "pengelolaan sampah"],
        "pertanian": ["tanaman", "peternakan", "lahan pertanian", "irigasi", "pupuk", "pestisida", "perlindungan tanaman"],
        "kelautan dan perikanan": ["perikanan", "kelautan", "kemaritiman"],
        "ketenagakerjaan": ["tenaga kerja", "kepegawaian", "cipta kerja", "hukuman disiplin", "atribut", "upah", "aparatur sipil negara", "penerimaan calon praja", "IPDN", "pakaian dinas", "pegawai negeri sipil", "keamanan kerja", "hak-hak pekerja", "serikat pekerja", "gaji", "perlindungan sosial"],
        "perpajakan": ["pajak penghasilan", "pajak pertambahan nilai", "tarif pajak", "penghindaran pajak", "insentif pajak"],
        "investasi": ["pasar modal", "saham", "obligasi", "regulasi investasi", "perlindungan investor", "modal ventura"],
        "transportasi": ["angkutan","jalan", "transportasi umum", "kendaraan bermotor", "bandara", "pelabuhan", "transportasi massal"],
        "keuangan": ["perbankan", "keuangan", "pembayaran", "pembiayaan", "asuransi", "pasar keuangan", "regulasi keuangan", "inflasi", "suku bunga", "anggaran", "anggaran pendapatan", "belanja daerah", "belanja", "harga", "standar harga", "pengelolaan keuangan daerah", "standar harga satuan", "keuangan daerah", "satuan biaya", "harga pasar", "kebijakan moneter"],
        "hak dan kewajiban": ["hak asasi manusia", "hak", "perlindungan perempuan", "pemberdayaan perempuan", "perlindungan anak", "gender", "kesetaraan gender", "pengarusutamaan gender", "responsif gender", "kesenjangan gender", "analisis gender", "perspektif gender", "korban kekerasan", "diskriminasi", "hak untuk hidup", "kebebasan berbicara", "hak beragama", "kepemilikan properti", "paten"],
        "komunikasi": ["komunikasi", "siaran", "penyiaran", "radio", "iklan", "dokumenter", "film"],
        "budaya dan pariwisata": ["desa wisata", "wisata", "pariwisata", "budaya", "kebudayaan"],
        "riset dan teknologi": ["riset dan teknologi", "teknologi", "inovasi"],
        "bencana alam": ["gempa bumi", "gelombang pasang", "gunung meletus", "tsunami", "erupsi", "banjir", "angin puting beliung", "tornado", "kekeringan", "tanah longsor"],
        "ekonomi": ["ekonomi", "industri", "perdagangan", "perdagangan bebas", "produksi barang"],
        "olahraga": ["pemuda dan olahraga", "kejuaraan olahraga", "olahragawan", "keolahragaan", "atlet", "pelatih olahraga", "olahraga"]
    }


    frekuensi_topik = {topik: 0 for topik in topik_kata_kunci}

    for topik, kata_kunci in topik_kata_kunci.items():
        for kata in kata_kunci:
            frekuensi_topik[topik] += text.count(kata)

    topik_peraturan_text = max(frekuensi_topik, key=frekuensi_topik.get)

    if frekuensi_topik[topik_peraturan_text] == 0:
        topik_peraturan_text = "Topik tidak ditemukan"
    
    judul =  extract_title(text)

    struktur_peraturan_pattern = {
    "Judul": judul,
    "Pembukaan": r'(Dengan Rahmat Tuhan Yang Maha Esa.*?)(?=Menimbang)',
    "Batang Tubuh": konten_penimbang + "\n\n" + peraturan_terkait + "\n\n" + konten_peraturan,
    "Penjelasan": r'(Penjelasan atas.*?)(?=Ditetapkan Di)',
    "Penutup": r'(?:Ditetapkan Di).*?(?=Lampiran|$)',
    "Lampiran": r'(Lampiran.*?$)'
}

    struktur_peraturan = {}

    for bagian, pola in struktur_peraturan_pattern.items():
        if bagian == "Batang Tubuh":
            struktur_peraturan[bagian] = pola
        else:
            matches = re.findall(pola, text, re.DOTALL | re.IGNORECASE)
            if matches:
                if bagian == "Lampiran":
                    struktur_peraturan[bagian] = matches[-1].strip()
                else:
                    struktur_peraturan[bagian] = matches[0].strip()
            else:
                struktur_peraturan[bagian] = "Tidak ditemukan"

    struktur_peraturan_mix = "\n\n".join([f"{bagian}: {konten}" for bagian, konten in struktur_peraturan.items()])

    return {
        "Pemrakarsa": pemrakarsa,
        "Level Peraturan": level_peraturan,
        "Konten Penimbang": konten_penimbang,
        "Peraturan Terkait": peraturan_terkait,
        "Konten Peraturan": konten_peraturan,
        "Kategori Peraturan": kategori_peraturan,
        "Topik Peraturan": topik_peraturan_text,
        "Struktur Peraturan": struktur_peraturan_mix
    }
 
def calculate_similarity(documents):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    combined_contents = [
        {
            'title': doc.title,
            'pemrakarsa': doc.pemrakarsa,
            'level_peraturan': doc.level_peraturan,
            'konten_penimbang': doc.konten_penimbang,
            'peraturan_terkait': doc.peraturan_terkait,
            'konten_peraturan': doc.konten_peraturan,
            'kategori_peraturan': doc.kategori_peraturan,
            'topik_peraturan': doc.topik_peraturan,
            'struktur_peraturan': doc.struktur_peraturan
        }
        for doc in documents
    ]

    field_vectors = {
        field: model.encode([doc[field] for doc in combined_contents])
        for field in combined_contents[0]
    }

    titles = [doc.title for doc in documents]
    similarity_results = []
    num_documents = len(titles)

    similarity_matrix = np.zeros((num_documents, num_documents))

    for i in range(num_documents):
        for j in range(i + 1, num_documents):
            total_similarity = 0
            detail_similarity = {}

            for field, vectors in field_vectors.items():
                score = cosine_similarity([vectors[i]], [vectors[j]])[0][0] * 100
                detail_similarity[field] = score
                total_similarity += score

            average_similarity = total_similarity / len(field_vectors)
            similarity_matrix[i, j] = average_similarity
            similarity_matrix[j, i] = average_similarity  # Matriks simetri

            similarity_results.append({
                'dokumen1': titles[i],
                'dokumen2': titles[j],
                'keterkaitan': f"{average_similarity:.2f}%",
                'detail_similarity': detail_similarity,
                'detail_url': f"/similarity_detail/{i}/{j}/"
            })

    return similarity_results, similarity_matrix

def perform_clustering(similarity_matrix, num_clusters=2):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(similarity_matrix)
    
    silhouette_avg = silhouette_score(similarity_matrix, labels)
    
    return silhouette_avg, labels

def calculate_component_similarity(dok_1, dok_2, komponen):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    try:
        dok_1_komponen = DokumenKomponen.objects.get(dokumen=dok_1, komponen=komponen).isi_komponen
        dok_2_komponen = DokumenKomponen.objects.get(dokumen=dok_2, komponen=komponen).isi_komponen
    except DokumenKomponen.DoesNotExist:
        return 0.0  # Jika tidak ada komponen, tidak ada keterkaitan
    
    embeddings = model.encode([dok_1_komponen, dok_2_komponen])
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0] * 100
    
    return similarity_score


