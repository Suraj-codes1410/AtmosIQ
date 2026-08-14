import { fetchApi } from './client';
import {
  DecisionSupportResponse,
  PredictionResponse,
  AttributionResponse,
  EnvironmentalValidationResponse,
  CounterfactualResponse,
  HealthResponse,
  ModelInfoResponse,
  EventResponse,
  EventCatalogItem
} from '../types';

export const getHealth = (): Promise<HealthResponse> =>
  fetchApi<HealthResponse>('/api/v1/health');

export const getModelInfo = (): Promise<ModelInfoResponse> =>
  fetchApi<ModelInfoResponse>('/api/v1/model/info');

export const getPrediction = (date: string): Promise<PredictionResponse> =>
  fetchApi<PredictionResponse>(`/api/v1/prediction/${date}`);

export const getAttribution = (date: string): Promise<AttributionResponse> =>
  fetchApi<AttributionResponse>(`/api/v1/attribution/${date}`);

export const getValidation = (date: string): Promise<EnvironmentalValidationResponse> =>
  fetchApi<EnvironmentalValidationResponse>(`/api/v1/validation/${date}`);

export const getCounterfactual = (date: string, scenario: string): Promise<CounterfactualResponse> =>
  fetchApi<CounterfactualResponse>(`/api/v1/counterfactual/${date}/${scenario}`);

export const getDecisionSupport = (date: string): Promise<DecisionSupportResponse> =>
  fetchApi<DecisionSupportResponse>(`/api/v1/decision-support/${date}`);

export const getEvents = (): Promise<EventCatalogItem[]> =>
  fetchApi<EventCatalogItem[]>('/api/v1/events');

export const getEventDetail = (eventId: string): Promise<EventResponse> =>
  fetchApi<EventResponse>(`/api/v1/events/${eventId}`);

export const getPeriodAnalysis = (startDate: string, endDate: string): Promise<DecisionSupportResponse[]> =>
  fetchApi<DecisionSupportResponse[]>(`/api/v1/event-analysis/${startDate}/${endDate}`);
