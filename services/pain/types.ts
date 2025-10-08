// Pain Detection System Types
export type PainEvent = {
  id: string; 
  time: string; 
  source: 'issues'|'forum'|'reviews'|'x'|'support'|'other';
  url?: string; 
  text: string; 
  eng: number; 
  org?: string;
};

export type PainRecord = PainEvent & { 
  pain: number; 
  severity: number; 
  clusterId?: string;
};

export type PainCluster = { 
  id: string; 
  label: string; 
  members: string[]; 
  avgPain: number; 
  count: number; 
  lastSeen: string;
};

export type PainSummary = { 
  last7: number; 
  prev21: number; 
  burst: number; 
  topOrgs: {org:string; count:number}[];
};