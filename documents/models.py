from django.db import models

class Document(models.Model):
    title = models.CharField(max_length=255)
    pemrakarsa = models.TextField()
    level_peraturan = models.TextField()
    konten_penimbang = models.TextField()
    peraturan_terkait = models.TextField()
    konten_peraturan = models.TextField()
    kategori_peraturan = models.TextField()
    topik_peraturan = models.TextField()
    struktur_peraturan = models.TextField()

    def __str__(self):
        return self.title



class Dokumen(models.Model):
    title = models.CharField(max_length=255)
    penandatangan = models.CharField(max_length=255)
    tanggal_ditandatangani = models.CharField(max_length=255)
    lembaga_mengeluarkan = models.CharField(max_length=255)
    no_dokumen = models.CharField(max_length=100)
    status_dokumen = models.CharField(max_length=50)
    lokasi_penerbit = models.CharField(max_length=255)
    sumber = models.TextField()
    ringkasan = models.TextField()

    def __str__(self):
        return self.title

class Komponen(models.Model):
    nama_komponen = models.CharField(max_length=255)

    def __str__(self):
        return self.nama_komponen

class DokumenKomponen(models.Model):
    dokumen = models.ForeignKey(Dokumen, on_delete=models.CASCADE)
    komponen = models.ForeignKey(Komponen, on_delete=models.CASCADE)
    isi_komponen = models.TextField()

    def __str__(self):
        return f"{self.dokumen.title} - {self.komponen.nama_komponen}"

class KeterkaitanDokumen(models.Model):
    dok_1 = models.ForeignKey(Dokumen, on_delete=models.CASCADE, related_name='keterkaitan_dok_1')
    dok_2 = models.ForeignKey(Dokumen, on_delete=models.CASCADE, related_name='keterkaitan_dok_2')
    nilai_total_keterkaitan = models.FloatField()

    def __str__(self):
        return f"Keterkaitan {self.dok_1.title} & {self.dok_2.title}"

class KeterkaitanKomponen(models.Model):
    keterkaitan_dokumen = models.ForeignKey(KeterkaitanDokumen, on_delete=models.CASCADE)
    komponen = models.ForeignKey(Komponen, on_delete=models.CASCADE)
    nilai_keterkaitan_komponen = models.FloatField()

    def __str__(self):
        return f"Keterkaitan {self.komponen.nama_komponen} dalam {self.keterkaitan_dokumen}"
