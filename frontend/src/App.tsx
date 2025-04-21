import React, { useState, ChangeEvent, useEffect } from 'react';
import { Container, Typography, Button, Box, Stepper, Step, StepLabel, LinearProgress } from '@mui/material';
import axios from 'axios';

const steps = [
  'Uploadファイル',
  '正規化・埋め込み',
  'DB挿入',
  'スコアリング',
  'Excel出力',
  '完了',
];

const App: React.FC = () => {
  const [fileA, setFileA] = useState<File | null>(null);
  const [fileB, setFileB] = useState<File | null>(null);
  const [downloadUrl, setDownloadUrl] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('');
  const [messages, setMessages] = useState<string[]>([]);
  const [activeStep, setActiveStep] = useState<number>(0);

  const handleFileAChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) setFileA(e.target.files[0]);
  };
  const handleFileBChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) setFileB(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!fileA || !fileB) return;
    setLoading(true);
    const form = new FormData();
    form.append('fileA', fileA);
    form.append('fileB', fileB);
    try {
      const { data } = await axios.post('http://localhost:8000/upload', form);
      const id = data.job_id;
      setJobId(id);
      setActiveStep(1);
      // SSEで進捗取得
      const es = new EventSource(`http://localhost:8000/status/stream/${id}`);
      es.addEventListener('progress', (e: MessageEvent) => {
        const d = JSON.parse(e.data);
        setMessages(prev => [...prev, d.step]);
        // ステップ判定
        if (/Cleared existing data/.test(d.step)) setActiveStep(0);
        else if (/Total embed/.test(d.step)) setActiveStep(1);
        else if (/CustomerStdA insert time/.test(d.step)) setActiveStep(2);
        else if (/Score calculation/.test(d.step) || /MatchCandidate/.test(d.step)) setActiveStep(3);
        else if (/Excel written/.test(d.step)) setActiveStep(4);
      });
      es.addEventListener('end', (e: MessageEvent) => {
        const d = JSON.parse(e.data);
        setStatus(d.status);
        if (d.download_url) setDownloadUrl(d.download_url);
        setActiveStep(5);
        setLoading(false);
        es.close();
      });
    } catch (err) {
      console.error(err);
      alert('Upload failed');
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 4 }}>
      <Typography variant="h5" gutterBottom>顧客マスター マッチング</Typography>
      <Box mb={2}>
        <input type="file" accept=".xlsx" onChange={handleFileAChange} />
      </Box>
      <Box mb={2}>
        <input type="file" accept=".xlsx" onChange={handleFileBChange} />
      </Box>
      <Button variant="contained" onClick={handleUpload} disabled={loading}>
        {loading ? 'Uploading...' : 'Upload'}
      </Button>

      {loading && (
        <Box mt={2}>
          <Stepper activeStep={activeStep}>
            {steps.map((label, index) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </Stepper>
          <LinearProgress variant="determinate" value={(activeStep / steps.length) * 100} />
          {messages.map((m, idx) => (
            <Typography key={idx} variant="body2">・ {m}</Typography>
          ))}
        </Box>
      )}
      {downloadUrl && (
        <Box mt={4}>
          <Button variant="outlined" href={downloadUrl} target="_blank">
            Download Result
          </Button>
        </Box>
      )}
    </Container>
  );
};

export default App;
