import React from 'react';
import { Menu, Activity } from 'lucide-react';

const Header = ({ onToggleSidebar, sidebarVisible }) => {
  return (
    <header className="bg-light-50 border-b border-light-400 px-4 py-3 flex items-center justify-between">
      <div className="flex items-center space-x-3">
        <button
          onClick={onToggleSidebar}
          className="p-2 hover:bg-light rounded-lg transition-colors"
          title={sidebarVisible ? 'Hide sidebar' : 'Show sidebar'}
        >
          <Menu className="w-5 h-5 text-muted" />
        </button>
        
        <div className="flex items-center space-x-2">
          <Activity className="w-6 h-6 text-primary" />
          <h1 className="text-xl font-semibold text-dark">
            Clinical Agent
          </h1>
        </div>
      </div>
      
      <div className="text-sm text-muted">
        Medical AI Assistant
      </div>
    </header>
  );
};

export default Header;
