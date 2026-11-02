from pathlib import Path
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions


def baca_pdf(path):
    reader = PdfReader(path)
    return "\n".join((p.extract_text() or "") for p in reader.pages)

def potong(teks, ukuran=1000, tumpang=200):
    """Pemotongan ukuran tetap dengan overlap (versi paling sederhana)."""
    hasil, mulai = [], 0
    while mulai < len(teks):
        hasil.append(teks[mulai:mulai + ukuran])
        mulai += ukuran - tumpang
    return [h for h in hasil if h.strip()]

# model embedding multilingual (jalan lokal, tanpa API key)
emb = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2")

client = chromadb.PersistentClient(path="./chroma_db")   # tersimpan di folder
koleksi = client.get_or_create_collection(
    name="pedoman", embedding_function=emb,
    metadata={"hnsw:space": "cosine"})                    # ukuran jarak: cosine

for pdf in Path("dokumen").glob("*.pdf"):
    chunks = potong(baca_pdf(pdf))
    koleksi.upsert(                                           # upsert: aman dijalankan ulang
        ids=[f"{pdf.stem}-{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[{"sumber": pdf.name, "chunk": i} for i in range(len(chunks))],
    )
    print(pdf.name, "→", len(chunks), "chunk")

hasil = koleksi.query(query_texts=["Berapa maksimal halaman bagian inti proposal PKM-RE?",], n_results=3, include=["documents", "metadatas", "distances"])

for doc, meta, jarak in zip(hasil["documents"][0], hasil["metadatas"][0], hasil["distances"][0]):
    print(f"[{meta['sumber']} · chunk {meta['chunk']}]  jarak={jarak:.3f}")
    print(doc[:200], "...\n")


UJI = [
    ("Berapa maksimal halaman bagian inti proposal PKM-RE?", "Bagian inti, maksimum 10 (sepuluh) halaman, Bab 1 Pendahuluan sampai dengan Daftar Pustaka"),
    ("Berapa dana yang diberikan Direktorat untuk proposal PKM yang lolos pendanaan?", "Rp. 6.000.000,00, Rp. 8.000.000,00, Direktorat Pembelajaran dan Kemahasiswaan, passing grade"),
    ("Berapa maksimum dana pendamping yang wajib diberikan perguruan tinggi?","Perguruan Tinggi wajib, maksimum, Rp. 2.000.000,00, tunai dan/atau barang"),
    ("Berapa jumlah anggota tim pengusul PKM-RE dan bagaimana susunannya?","3-5 mahasiswa aktif, 1 mahasiswa sebagai ketua, 2-4 mahasiswa sebagai anggota"),
    ("Apa objek riset PKM-RE?","Objek riset PKM-RE, fenomena alamiah, fisika, kimia, biologi dan matematika"),
    ("Apa objek riset PKM-RE dan bidang apa yang digabungkan?","gabungan antara bidang Kedokteran dan Kesehatan,Farmasi,Pertanian,Teknologi,Ilmu Dasar dan Matematika,Sains Material (Material Science),Kebumian"),
    ("Apa saja luaran wajib PKM-RE?","Luaran wajib PKM-RE, Laporan kemajuan, Laporan akhir, Artikel ilmiah, Akun media sosial"),
    ("Berapa batas maksimum indeks similaritas proposal?","Turnitin, iThenticate, indeks similaritas maksimum 25%"),
    ("Bagaimana rumus Nilai Akhir PKP2 untuk PKM-RE dan PKM-RSH?","PKM 6 Bidang, NA = 0,3*NP + 0,2*NLK+0,5*NM, Nilai Laporan Kemajuan, PKP2"),
    ("Kapan jadwal pengiklanan serentak di media sosial dan apa kontennya?","Sabtu, 23 Mei 2026, Sabtu, 20 Juni 2026, Sabtu, 8 Agustus 2026, Pengenalan Program, Hasil Program PKM"),
    ("Berapa unggahan yang wajib diberi Ads dan berapa batas dananya?,","3 (tiga) unggahan, Ads (iklan berbayar), Rp500.000,00"),
    ("Apa saja aspek etika yang wajib dicantumkan di metode PKM-RE?","Etika Terhadap Subjek Penelitian (Jika Menggunakan Makhluk Hidup), Kesehatan dan Keselamatan Kerja (K3) Laboratorium, Manajemen Limbah dan Dampak Lingkungan, Integritas Data dan Anti-Plagiarisme"),
    ("Persetujuan apa yang wajib diusulkan jika PKM-RE memakai hewan coba atau bahan tanaman?","hewan coba, kode etik penggunaan hewan, komite etik, determinasi tanaman"),
    ("Berapa batas maksimum tiap jenis pengeluaran di rekapitulasi RAB?","Bahan habis pakai, maksimum 60%, maksimum 15%, Transportasi lokal maksimum 30%, Rekapitulasi"),
    ("Berapa proposal yang boleh diusulkan satu mahasiswa?","Setiap mahasiswa hanya dapat mengusulkan 1 (satu) proposal PKM, baik sebagai ketua atau anggota")
]

def hit_at_k(uji, k=3):
    benar = 0
    for pertanyaan, kunci in uji:
        docs = koleksi.query(query_texts=[pertanyaan], n_results=k)["documents"][0]
        if any(kunci.lower() in d.lower() for d in docs):
            benar += 1
    return benar / len(uji)

for k in (1, 3, 5):
    print(f"Hit@{k} = {hit_at_k(UJI, k):.2f}")


