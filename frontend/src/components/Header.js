import React from 'react';
import { Menu, ShieldCheck } from 'lucide-react';

const Header = ({ onToggleSidebar, sidebarVisible }) => {
  return (
    <header className="bg-white border-b border-slate-200 px-5 py-3 flex items-center justify-between">
      <div className="flex items-center space-x-3">
        <button
          onClick={onToggleSidebar}
          className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
          title={sidebarVisible ? 'Hide sidebar' : 'Show sidebar'}
        >
          <Menu className="w-5 h-5 text-slate-500" />
        </button>

        <div className="flex items-center space-x-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-teal-500 to-teal-700 flex items-center justify-center shadow-sm">
            <ShieldCheck className="w-4.5 h-4.5 text-white" strokeWidth={2.5} />
          </div>
          <div>
            <h1 className="text-lg font-bold text-slate-900 leading-tight tracking-tight">
              SafeToSay
            </h1>
            <p className="text-[10px] text-slate-400 font-medium -mt-0.5 tracking-wider uppercase">
              Clinical AI Agent
            </p>
          </div>
        </div>
      </div>

      <div className="flex items-center space-x-3">
        <div className="hidden sm:flex items-center space-x-1.5 bg-teal-50 text-teal-700 px-3 py-1.5 rounded-full text-xs font-medium">
          <span className="w-1.5 h-1.5 rounded-full bg-teal-500 animate-pulse" />
          <span>Safety Gate Active</span>
        </div>
      </div>
    </header>
  );
};

export default Header;
