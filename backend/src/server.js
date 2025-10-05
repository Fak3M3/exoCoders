import express from 'express';
import cors from 'cors';
import multer from 'multer';

const app = express();
const upload = multer({ dest: 'uploads/' });

app.use(cors());

app.post('/predict', upload.single('file'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ success: false, error: 'No se envió archivo' });
  }

  res.json({
    success: true,
    prediction: {
      mensaje: 'Archivo procesado correctamente',
      nombreArchivo: req.file.originalname,
      resultado: 'Exoplaneta detectado'
    }
  });
});

const PORT = 5000;
app.listen(PORT, () => {
  console.log(`Servidor Express corriendo en http://localhost:${PORT}`);
});