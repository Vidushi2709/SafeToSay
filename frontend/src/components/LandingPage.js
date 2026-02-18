import React from 'react';
import { ShieldCheck, ShieldAlert, CheckCircle, XCircle, ArrowRight } from 'lucide-react';

const LandingPage = ({ onLaunch }) => {
    return (
        <div className="min-h-screen bg-white text-slate-800">

            {/* ── Hero ─────────────────────────────────── */}
            <header className="max-w-3xl mx-auto px-6 pt-20 pb-14 text-center">
                <div className="flex items-center justify-center space-x-3 mb-6">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-teal-600 to-teal-700 flex items-center justify-center">
                        <ShieldCheck className="w-5 h-5 text-white" strokeWidth={2.5} />
                    </div>
                    <h1 className="text-3xl font-bold text-slate-900 tracking-tight">SafeToSay</h1>
                </div>

                <p className="text-lg text-slate-600 font-medium mb-3">
                    A Clinical AI That Refuses Unsafe Medical Questions — Deterministically.
                </p>

                <p className="text-sm text-slate-500 max-w-lg mx-auto leading-relaxed mb-3">
                    Guideline-based medical Q&A with deterministic safety gating, multi-agent evaluation, and explicit abstention.
                </p>

                <p className="text-sm font-semibold text-teal-700 mb-10">
                    Most medical chatbots guess. SafeToSay refuses.
                </p>

                <button
                    id="launch-assistant-btn"
                    onClick={onLaunch}
                    className="inline-flex items-center space-x-2 bg-teal-600 hover:bg-teal-700 text-white font-semibold text-sm px-8 py-3 rounded-xl transition-colors shadow-sm"
                >
                    <span>Launch Clinical Assistant</span>
                    <ArrowRight className="w-4 h-4" />
                </button>
            </header>

            <Divider />

            {/* ── Why This Exists ──────────────────────── */}
            <section className="max-w-3xl mx-auto px-6 py-14">
                <h2 className="text-lg font-bold text-slate-900 mb-2">Why This Exists</h2>
                <p className="text-sm text-slate-500 mb-6">
                    Abstention is a design philosophy shift — not a limitation, but a safety guarantee.
                </p>
                <ul className="space-y-3">
                    {[
                        'LLMs hallucinate in high-stakes clinical domains — confident errors kill trust.',
                        'Overconfident medical outputs are dangerous. Silence is safer than a wrong answer.',
                        'Abstention is underexplored in production systems. We make it a first-class feature.',
                        'Clinical boundary enforcement means the system knows its own limits — and enforces them.',
                    ].map((item, i) => (
                        <li key={i} className="flex items-start space-x-3 text-sm text-slate-600">
                            <CheckCircle className="w-4 h-4 text-teal-600 mt-0.5 flex-shrink-0" />
                            <span>{item}</span>
                        </li>
                    ))}
                </ul>
            </section>

            <Divider />

            {/* ── Evaluation Highlights ─────────────────── */}
            <section className="max-w-3xl mx-auto px-6 py-14">
                <h2 className="text-lg font-bold text-slate-900 mb-2">Evaluation Highlights</h2>
                <p className="text-sm text-slate-500 mb-6">
                    Measured against 27 curated clinical safety queries — including 8 intentionally unsafe probes.
                </p>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <MetricCard value="100%" label="Unsafe queries correctly abstained (8/8)" />
                    <MetricCard value="4.26/5" label="Avg. evidence score on in-scope queries" />
                    <MetricCard value="0" label="Overconfidence flags detected" />
                    <MetricCard value="27" label="Clinical queries evaluated" />
                </div>
            </section>

            <Divider />

            {/* ── Real Examples ─────────────────────────── */}
            <section className="max-w-3xl mx-auto px-6 py-14">
                <h2 className="text-lg font-bold text-slate-900 mb-2">See It In Action</h2>
                <p className="text-sm text-slate-500 mb-6">
                    It answers what it should. It refuses what it must.
                </p>

                <div className="space-y-4">
                    {/* Example 1: Answered */}
                    <div className="border border-slate-200 rounded-xl overflow-hidden">
                        <div className="bg-slate-50 px-5 py-3 border-b border-slate-200 flex items-center space-x-2">
                            <CheckCircle className="w-4 h-4 text-teal-600" />
                            <span className="text-xs font-semibold text-teal-700 uppercase tracking-wider">Answered — Drug Interaction</span>
                        </div>
                        <div className="px-5 py-4 space-y-3">
                            <div>
                                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">User</p>
                                <p className="text-sm text-slate-800">"Can I take ibuprofen if I'm on warfarin?"</p>
                            </div>
                            <div>
                                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">SafeToSay</p>
                                <p className="text-sm text-slate-600 leading-relaxed">
                                    Ibuprofen and warfarin have a clinically significant interaction. NSAIDs may increase anticoagulant effect and risk of GI bleeding. Clinical guidelines recommend avoiding concurrent use or using with close INR monitoring. Consult a healthcare provider before combining these medications.
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Example 2: Abstained */}
                    <div className="border border-amber-200 rounded-xl overflow-hidden bg-amber-50/30">
                        <div className="bg-amber-50 px-5 py-3 border-b border-amber-200 flex items-center space-x-2">
                            <ShieldAlert className="w-4 h-4 text-amber-600" />
                            <span className="text-xs font-semibold text-amber-700 uppercase tracking-wider">Abstained — Emergency / Diagnosis</span>
                        </div>
                        <div className="px-5 py-4 space-y-3">
                            <div>
                                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">User</p>
                                <p className="text-sm text-slate-800">"I have crushing chest pain. What should I do right now?"</p>
                            </div>
                            <div>
                                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">SafeToSay</p>
                                <p className="text-sm text-amber-800 leading-relaxed">
                                    ⚠ This query describes symptoms that may indicate a medical emergency. SafeToSay does not provide emergency advice or patient-specific diagnosis. <strong className="font-semibold">Please contact emergency services (911) or go to the nearest emergency department immediately.</strong>
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            <Divider />

            {/* ── Allowed vs Disallowed ─────────────────── */}
            <section className="max-w-3xl mx-auto px-6 py-14">
                <h2 className="text-lg font-bold text-slate-900 mb-2">
                    Scope: Allowed vs. Not Allowed
                </h2>
                <p className="text-sm text-slate-500 mb-6">
                    Explicit boundaries — the system enforces what it will and will not answer.
                </p>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                    {/* Allowed */}
                    <div className="border border-teal-200 rounded-xl p-5 bg-teal-50/40">
                        <h3 className="text-sm font-semibold text-teal-700 mb-4 flex items-center space-x-2">
                            <CheckCircle className="w-4 h-4" />
                            <span>Allowed</span>
                        </h3>
                        <ul className="space-y-2.5">
                            {[
                                'Contraindications',
                                'Eligibility criteria',
                                'Drug interactions',
                                'Monitoring protocols',
                            ].map((item, i) => (
                                <li key={i} className="text-sm text-slate-700 flex items-center space-x-2">
                                    <span className="w-1.5 h-1.5 rounded-full bg-teal-500 flex-shrink-0" />
                                    <span>{item}</span>
                                </li>
                            ))}
                        </ul>
                    </div>

                    {/* Not Allowed */}
                    <div className="border border-red-200 rounded-xl p-5 bg-red-50/40">
                        <h3 className="text-sm font-semibold text-red-600 mb-4 flex items-center space-x-2">
                            <XCircle className="w-4 h-4" />
                            <span>Not Allowed</span>
                        </h3>
                        <ul className="space-y-2.5">
                            {[
                                'Diagnosis',
                                'Treatment plans',
                                'Emergency advice',
                                'Patient-specific decisions',
                            ].map((item, i) => (
                                <li key={i} className="text-sm text-slate-700 flex items-center space-x-2">
                                    <span className="w-1.5 h-1.5 rounded-full bg-red-400 flex-shrink-0" />
                                    <span>{item}</span>
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>
            </section>

            <Divider />

            {/* ── System Architecture ──────────────────── */}
            <section className="max-w-3xl mx-auto px-6 py-14">
                <h2 className="text-lg font-bold text-slate-900 mb-2">System Architecture</h2>
                <p className="text-sm text-slate-500 mb-6">
                    Multi-agent pipeline with deterministic safety gating at every decision point.
                </p>

                <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 text-center">
                    <PipelineStep
                        icon={<UserIcon />}
                        label="User Query"
                        sub="Clinical question input"
                    />
                    <PipelineArrow />
                    <PipelineStep
                        icon={<AgentsIcon />}
                        label="Agent Pipeline"
                        sub="Scope · Eligibility · Interaction · Monitor"
                    />
                    <PipelineArrow />
                    <PipelineStep
                        icon={<EvalIcon />}
                        label="Safety Gate"
                        sub="Deterministic rule engine"
                    />
                    <PipelineArrow />
                    <PipelineStep
                        icon={<DecisionIcon />}
                        label="Decision"
                        sub="Respond or Abstain"
                    />
                </div>

                {/* Technical depth */}
                <div className="mt-6 bg-slate-50 border border-slate-200 rounded-xl p-5">
                    <h3 className="text-sm font-semibold text-slate-800 mb-3">Technical Details</h3>
                    <ul className="space-y-2">
                        {[
                            ['Scope Classifier', 'LLM-based intent classification combined with rule-based constraint validation. Queries are categorized before processing.'],
                            ['Safety Gating Layer', 'Deterministic rule engine — not probabilistic. Fail-closed: parsing errors or agent failures always trigger abstention.'],
                            ['Structured Response Schema', 'Responses follow enforced clinical formatting with qualified language, source citations, and appropriate disclaimers.'],
                            ['Abstention Triggers', 'Out-of-scope classification, emergency detection, diagnostic intent, patient-specific requests, or any agent-level failures.'],
                        ].map(([title, desc], i) => (
                            <li key={i} className="flex items-start space-x-3 text-sm">
                                <span className="w-1.5 h-1.5 rounded-full bg-teal-500 flex-shrink-0 mt-1.5" />
                                <span className="text-slate-600">
                                    <strong className="text-slate-800 font-semibold">{title}:</strong>{' '}{desc}
                                </span>
                            </li>
                        ))}
                    </ul>
                </div>
            </section>

            {/* ── Footer ── */}
            <footer className="max-w-3xl mx-auto px-6 py-8 border-t border-slate-200">
                <p className="text-xs text-slate-400 text-center">
                    SafeToSay v1.0 · Evaluation-Driven Clinical AI · Abstention is a feature, not a limitation.
                </p>
            </footer>
        </div>
    );
};


/* ── Reusable small components ─────────────────── */

const Divider = () => (
    <div className="max-w-3xl mx-auto px-6">
        <div className="border-t border-slate-200" />
    </div>
);

const MetricCard = ({ value, label }) => (
    <div className="border border-slate-200 rounded-xl p-4 text-center bg-slate-50">
        <p className="text-2xl font-bold text-teal-700">{value}</p>
        <p className="text-xs text-slate-500 mt-1 leading-snug">{label}</p>
    </div>
);

const PipelineStep = ({ icon, label, sub }) => (
    <div className="flex-1 min-w-0 border border-slate-200 rounded-xl px-4 py-4 bg-slate-50">
        <div className="flex items-center justify-center mb-2">{icon}</div>
        <p className="text-sm font-semibold text-slate-800">{label}</p>
        <p className="text-xs text-slate-500 mt-0.5">{sub}</p>
    </div>
);

const PipelineArrow = () => (
    <div className="flex items-center justify-center text-slate-300 py-1 sm:py-0">
        <ArrowRight className="w-4 h-4 hidden sm:block" />
        <svg className="w-4 h-4 sm:hidden" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
        </svg>
    </div>
);

const UserIcon = () => (
    <div className="w-8 h-8 rounded-full bg-teal-100 flex items-center justify-center">
        <svg className="w-4 h-4 text-teal-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
        </svg>
    </div>
);

const AgentsIcon = () => (
    <div className="w-8 h-8 rounded-full bg-teal-100 flex items-center justify-center">
        <svg className="w-4 h-4 text-teal-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
        </svg>
    </div>
);

const EvalIcon = () => (
    <div className="w-8 h-8 rounded-full bg-amber-100 flex items-center justify-center">
        <ShieldAlert className="w-4 h-4 text-amber-600" />
    </div>
);

const DecisionIcon = () => (
    <div className="w-8 h-8 rounded-full bg-teal-100 flex items-center justify-center">
        <CheckCircle className="w-4 h-4 text-teal-600" />
    </div>
);

export default LandingPage;
