import React, { useState, FormEvent, ChangeEvent } from 'react';
import {
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  Box,
  Grid,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent
} from '@mui/material';
import axios from 'axios';

interface FormData {
  Gender: string;
  Married: string;
  Dependents: string;
  Education: string;
  Self_Employed: string;
  ApplicantIncome: string;
  CoapplicantIncome: string;
  LoanAmount: string;
  Loan_Amount_Term: string;
  Credit_History: string;
  Property_Area: string;
}

interface PredictionResponse {
  prediction: string;
  confidence?: number;
}

function App() {
  const [formData, setFormData] = useState<FormData>({
    Gender: '',
    Married: '',
    Dependents: '',
    Education: '',
    Self_Employed: '',
    ApplicantIncome: '',
    CoapplicantIncome: '',
    LoanAmount: '',
    Loan_Amount_Term: '',
    Credit_History: '',
    Property_Area: ''
  });

  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (event: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData({
      ...formData,
      [event.target.name]: event.target.value
    });
  };

  const handleSelectChange = (event: SelectChangeEvent<string>) => {
    setFormData({
      ...formData,
      [event.target.name]: event.target.value
    });
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await axios.post('/api/predict', formData);
      setPrediction(response.data);
    } catch (err) {
      setError('Failed to get prediction. Please check your input and try again.');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Loan Eligibility Predictor
        </Typography>
        <Typography variant="subtitle1" align="center" color="text.secondary" paragraph>
          Enter your details to check loan eligibility
        </Typography>

        <Box component="form" onSubmit={handleSubmit} sx={{ mt: 3 }}>
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Gender</InputLabel>
                <Select
                  name="Gender"
                  value={formData.Gender}
                  onChange={handleSelectChange}
                  required
                >
                  <MenuItem value="Male">Male</MenuItem>
                  <MenuItem value="Female">Female</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Married</InputLabel>
                <Select
                  name="Married"
                  value={formData.Married}
                  onChange={handleSelectChange}
                  required
                >
                  <MenuItem value="Yes">Yes</MenuItem>
                  <MenuItem value="No">No</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Dependents</InputLabel>
                <Select
                  name="Dependents"
                  value={formData.Dependents}
                  onChange={handleSelectChange}
                  required
                >
                  <MenuItem value="0">0</MenuItem>
                  <MenuItem value="1">1</MenuItem>
                  <MenuItem value="2">2</MenuItem>
                  <MenuItem value="3+">3+</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Education</InputLabel>
                <Select
                  name="Education"
                  value={formData.Education}
                  onChange={handleSelectChange}
                  required
                >
                  <MenuItem value="Graduate">Graduate</MenuItem>
                  <MenuItem value="Not Graduate">Not Graduate</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Self Employed</InputLabel>
                <Select
                  name="Self_Employed"
                  value={formData.Self_Employed}
                  onChange={handleSelectChange}
                  required
                >
                  <MenuItem value="Yes">Yes</MenuItem>
                  <MenuItem value="No">No</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6}>
              <TextField
                name="ApplicantIncome"
                label="Applicant Income"
                type="number"
                value={formData.ApplicantIncome}
                onChange={handleChange}
                fullWidth
                required
              />
            </Grid>

            <Grid item xs={12} sm={6}>
              <TextField
                name="CoapplicantIncome"
                label="Coapplicant Income"
                type="number"
                value={formData.CoapplicantIncome}
                onChange={handleChange}
                fullWidth
              />
            </Grid>

            <Grid item xs={12} sm={6}>
              <TextField
                name="LoanAmount"
                label="Loan Amount"
                type="number"
                value={formData.LoanAmount}
                onChange={handleChange}
                fullWidth
                required
              />
            </Grid>

            <Grid item xs={12} sm={6}>
              <TextField
                name="Loan_Amount_Term"
                label="Loan Amount Term (months)"
                type="number"
                value={formData.Loan_Amount_Term}
                onChange={handleChange}
                fullWidth
                required
              />
            </Grid>

            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Credit History</InputLabel>
                <Select
                  name="Credit_History"
                  value={formData.Credit_History}
                  onChange={handleSelectChange}
                  required
                >
                  <MenuItem value="1.0">Good</MenuItem>
                  <MenuItem value="0.0">Poor</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Property Area</InputLabel>
                <Select
                  name="Property_Area"
                  value={formData.Property_Area}
                  onChange={handleSelectChange}
                  required
                >
                  <MenuItem value="Urban">Urban</MenuItem>
                  <MenuItem value="Semiurban">Semiurban</MenuItem>
                  <MenuItem value="Rural">Rural</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12}>
              <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
                <Button
                  type="submit"
                  variant="contained"
                  size="large"
                  disabled={loading}
                  sx={{ minWidth: 200 }}
                >
                  {loading ? <CircularProgress size={24} /> : 'Predict Eligibility'}
                </Button>
              </Box>
            </Grid>
          </Grid>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mt: 3 }}>
            {error}
          </Alert>
        )}

        {prediction && (
          <Alert
            severity={prediction.prediction === 'Y' ? 'success' : 'warning'}
            sx={{ mt: 3 }}
          >
            <Typography variant="h6">
              {prediction.prediction === 'Y' ? 'Loan Approved!' : 'Loan Not Approved'}
            </Typography>
            {prediction.confidence && (
              <Typography variant="body2">
                Confidence: {(prediction.confidence * 100).toFixed(1)}%
              </Typography>
            )}
          </Alert>
        )}
      </Paper>
    </Container>
  );
}

export default App;